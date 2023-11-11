extern crate nalgebra as na;
use color_eyre::eyre::Result;

use na::{Vector3,convert, Isometry3,UnitQuaternion,Rotation3};
use std::collections::HashMap;
use vision::{Float,load_runtime_conf};
use vision::image::features::image_feature::ImageFeature;
use vision::sfm::{
    landmark::{euclidean_landmark::EuclideanLandmark,Landmark},
    runtime_parameters::RuntimeParameters,
    pnp::{pnp_config::PnPConfig, run_pnp}
};
use vision::sensors::camera::perspective::Perspective;
use vision::numerics::{loss, weighting, pose::from_matrix_3x4, lie};

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    //let scenario = "60_10"; //Maybe too little translation
    let scenario = "pnp";
    //let scenario = "trans_y";
    //let scenario = "trans_z";

    let dataset = "Suzanne";
    //let dataset = "sphere";
    //let dataset = "Cube";

    let cam_features_path = format!("{}/{}/camera_features_{}.yaml",runtime_conf.local_data_path,scenario,dataset);
    let landmark_path = format!("{}/{}/landmarks_{}.yaml",runtime_conf.local_data_path,scenario,dataset);

    let loaded_data_cam = models_cv::io::deserialize_feature_matches(&cam_features_path);
    let loaded_data_landmarks = models_cv::io::deserialize_landmarks(&landmark_path);


    let camera_map = loaded_data_cam.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let intrinsic_matrix = cf.get_intrinsic_matrix().cast::<Float>();
        (cam_id,Perspective::<Float>::from_matrix(&intrinsic_matrix, true))
    }).collect::<HashMap<_,_>>();

    let feature_map = loaded_data_cam.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let cam_height =  camera_map.get(&cam_id).expect("Invalid key!").get_cy()*2.0;
        let map = cf.get_feature_map();
        //Synthetic model data is defined with a RHS along -Z. Point are exported as-is, so we flip them here
        let feature_map = map.iter().map(|(k,v)| (*k,ImageFeature::new(v.x as Float, cam_height - 1.0 - (v.y as Float) ,Some(*k)))).collect::<HashMap<_,_>>();
        (cam_id,feature_map)
    }).collect::<HashMap<_,_>>();

    let camera_poses = loaded_data_cam.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let map = cf.get_view_matrix().cast::<Float>();
        (cam_id,map)
    }).collect::<HashMap<_,_>>();

    // poses are defined in Computer Graphis Coordinate Systems. We need to flip it to Computer Vision
    let change_of_basis_x = UnitQuaternion::from_scaled_axis(Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI).scaled_axis());

    let cam_id = 0;
    let camera = camera_map.get(&cam_id).unwrap();
    let feature_map = feature_map.get(&cam_id).unwrap();


    // Gt Pose
    let camera_pose_gt_matrix = camera_poses.get(&cam_id).unwrap();
    let camera_cam_world_cg = from_matrix_3x4(camera_pose_gt_matrix);
    let camera_cam_world_vision = change_of_basis_x*camera_cam_world_cg*change_of_basis_x;
    let camera_pose = Some(camera_cam_world_vision);

    let camera_pose = None;

    let landmark_map = loaded_data_landmarks.iter().map(|l| {
        let v = change_of_basis_x*convert::<Vector3<f32>,Vector3<Float>>(*l.get_position());
        let id = *l.get_id();
        (id,EuclideanLandmark::<Float>::from_state_with_id(v,&Some(id)))
    }).collect::<HashMap<_,_>>();
 
    let pnp_config = PnPConfig::new(camera, &landmark_map, feature_map, &camera_pose);
    //TODO: split runtime parameters to BA and PNP. A lot of params are not needed
    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![8e4 as usize; 1],
        eps: vec![1e-8],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 

        delta_eps: 1e-30,
        taus: vec![1.0e0],
        lm: true,
        debug: false,
        print: true,
        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {}),
        cg_threshold: 1e-6,
        cg_max_it: 2e3 as usize
    };

    let (optimized_state, state_debug_list) = run_pnp(&pnp_config,&runtime_parameters);

    let camera_positions = &optimized_state.get_camera_positions();
    let cam_pos = camera_positions[0];
    println!("Pos: {}",cam_pos.to_matrix());

    let state_serialized = serde_yaml::to_string(&optimized_state.to_serial());
    let debug_states_serialized = serde_yaml::to_string(&state_debug_list);

    std::fs::write(format!("{}/pnp.txt",runtime_conf.output_path), state_serialized?).expect("Unable to write file");


    Ok(())
}