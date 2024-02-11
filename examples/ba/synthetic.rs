extern crate nalgebra as na;
use color_eyre::eyre::Result;
use vision::sfm::landmark::{Landmark,inverse_depth_landmark::InverseLandmark, euclidean_landmark::EuclideanLandmark};

use std::collections::{HashMap, HashSet};
use vision::{Float,load_runtime_conf};
use vision::sfm::{
    triangulation::Triangulation,
    bundle_adjustment::{run_ba, ba_config::{BAConfig,conversions::compute_path_id_pairs}}, 
    epipolar::tensor::BifocalType,
    runtime_parameters::RuntimeParameters,
    bundle_adjustment::ba_config::filtering::filter_config,
    pnp::run_pnp
};
use vision::sensors::camera::perspective::Perspective;
use vision::image::features::{matches::Match,image_feature::ImageFeature};
use vision::numerics::{loss, weighting};
use na::{Rotation3,Isometry3,Vector3,UnitQuaternion};

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    let scenario = "60_10"; //Maybe too little translation
    //let scenario = "trans_x";
    //let scenario = "trans_y";
    //let scenario = "trans_z";
    //let dataset = "Suzanne";
    let dataset = "sphere";
    //let dataset = "Cube";

    let image_width = 640;
    let image_height = 480;

    let cam_features_path = format!("{}/{}/camera_features_{}.yaml",runtime_conf.local_data_path,scenario,dataset);

    let loaded_data = models_cv::io::deserialize_feature_matches(&cam_features_path);


    let camera_map = loaded_data.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let intrinsic_matrix = cf.get_intrinsic_matrix().cast::<Float>();
        (cam_id,Perspective::<Float>::from_matrix(&intrinsic_matrix, true))
    }).collect::<HashMap<_,_>>();

    let feature_map = loaded_data.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let map = cf.get_feature_map();
        (cam_id,map)
    }).collect::<HashMap<_,_>>();

    let camera_poses = loaded_data.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let map = cf.get_view_matrix().cast::<Float>();
        (cam_id,map)
    }).collect::<HashMap<_,_>>();

    //let camera_id_pairs = vec!((1,2));
    //let camera_id_pairs = vec!((0,1));
    //let camera_id_pairs = vec!((0,2));
    //let camera_id_pairs = vec!((0,1),(1,2));
    //let camera_id_pairs = vec!((0,1),(1,2),(2,3));
    let camera_id_pairs = vec!((0,1),(1,2),(2,3),(3,4)); //TODO: Investigate why this leads to spurious landmarks -> Maybe its the feature tracks
    //let camera_id_pairs = vec!((0,1),(1,2),(2,3),(3,4),(4,5),(5,6));

    let match_map = camera_id_pairs.iter().map(|(id1,id2)| {
        let fm_1 = feature_map.get(id1).expect("Feature map for cam id not available!");
        let fm_2 = feature_map.get(id2).expect("Feature map for cam id not available!");

        let keys_1 = fm_1.keys().into_iter().map(|&id| id).collect::<HashSet<_>>();
        let keys_2 = fm_2.keys().into_iter().map(|&id| id).collect::<HashSet<_>>();

        let shared_keys = keys_1.intersection(&keys_2).collect::<HashSet<_>>();

        let matches = shared_keys.iter().map(|&key| {
            let f_1 = fm_1.get(key).expect("Invalid key!");
            let f_2 = fm_2.get(key).expect("Invalid key!");
            let cam_1_height =  camera_map.get(id1).expect("Invalid key!").get_cy()*2.0;
            let cam_2_height =  camera_map.get(id2).expect("Invalid key!").get_cy()*2.0;

            assert!(cam_1_height.fract() == 0.0);
            assert!(cam_2_height.fract() == 0.0);

            //Synthetic model data is defined with a RHS along -Z. Point are exported as-is, so we flip them here
            let image_feature_1 = ImageFeature::new(f_1.x as Float, cam_1_height - 1.0 - (f_1.y as Float), Some(*key));
            let image_feature_2 = ImageFeature::new(f_2.x as Float, cam_2_height - 1.0 - (f_2.y as Float), Some(*key));

            Match::new(image_feature_1, image_feature_2)
        }).collect::<Vec<_>>();

        ((*id1,*id2),matches)

    }).collect::<HashMap<_,_>>();

    let paths = vec![camera_id_pairs.iter().map(|&(_,c)| c).collect::<Vec<_>>()];
    let root_id = camera_id_pairs[0].0;

    // poses are defined in Computer Graphis Coordinate Systems. We need to flip it to Computer Vision
    let change_of_basis = UnitQuaternion::from_scaled_axis(Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI).scaled_axis());
    let pose_map_gt = camera_id_pairs.iter().map(|(id1,id2)| {
        let p1 = camera_poses.get(id1).expect("Camera map for cam id not available");
        let p2 = camera_poses.get(id2).expect("Camera map for cam id not available");
        
        let rot_2 = Rotation3::from_matrix(&p2.fixed_view::<3,3>(0, 0).into_owned());
        let trans_2 = p2.column(3).into_owned();
        let iso_cam2_world_neg_z = Isometry3::new(trans_2,rot_2.scaled_axis());

        let rot_1 = Rotation3::from_matrix(&p1.fixed_view::<3,3>(0, 0).into_owned());
        let trans_1 = p1.column(3).into_owned();
        let iso_cam1_world_neg_z = Isometry3::new(trans_1,rot_1.scaled_axis());

        let pose_12 = change_of_basis*iso_cam1_world_neg_z*iso_cam2_world_neg_z.inverse()*change_of_basis;

        ((*id1,*id2),pose_12)
    }).collect::<HashMap<_,_>>();

    //let pose_map_gt_option = Some(pose_map_gt);
    let pose_map_gt_option = None;


    //TODO: Add GT Landmarks
    let mut sfm_config_fundamental = BAConfig::new(root_id, &paths, pose_map_gt_option , camera_map, &match_map, 
        BifocalType::FUNDAMENTAL, Triangulation::STEREO, 1.0, 3e0, false, image_width, image_height); 
    filter_config(&mut sfm_config_fundamental, 5e2, false, true, Triangulation::STEREO);
    
    let initial_z = sfm_config_fundamental.pose_map().get(&camera_id_pairs[0]).unwrap().translation.z;
    for (key, pose) in sfm_config_fundamental.pose_map().iter() {
        let translation = pose.translation;
        let z = translation.z;
        let scale = initial_z / z;
        println!("Key: {:?}, Pose: {:?}, Scale: {}", key, pose,scale);
    }

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

    let runtime_parameters_pnp = RuntimeParameters {
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




    //let trajectories = compute_path_id_pairs(sfm_config_fundamental.root(), sfm_config_fundamental.paths());

    let trajectories = vec!(vec!((0,1)));

    let (optimized_state, state_debug_list) = run_ba::<_,6,InverseLandmark<Float>,_,_>(&sfm_config_fundamental, &runtime_parameters, &trajectories);
    sfm_config_fundamental.update_state(&optimized_state);

    let cam_0_idx = optimized_state.get_camera_id_map().get(&0).unwrap();
    let cam_pos_0 = optimized_state.get_camera_positions()[*cam_0_idx];
    let landmarks = sfm_config_fundamental.landmark_map().get(&(0,1)).unwrap();
    let first_landmark = landmarks.first().unwrap();
    let first_landmark_id = first_landmark.get_id().unwrap();

    let world_cam_0 = sfm_config_fundamental.abs_pose_map().get(&0).unwrap().clone();
    let landmark_vec = optimized_state.get_landmarks();
    let landmark_world_as_vec = landmark_vec.iter().filter(|l| l.get_id().is_some()).filter(|l| l.get_id().unwrap() == first_landmark_id).collect::<Vec<_>>().clone();
    assert_eq!(landmark_world_as_vec.len(),1);
    let landmark_w = landmark_world_as_vec.first().unwrap();
    let landmark_rel = world_cam_0.inverse()*landmark_w.get_euclidean_representation();

    println!("Cam 0 config: {}", world_cam_0);
    println!("Cam 0 state first: {}", cam_pos_0);
    println!("Landmark config {} : {}", first_landmark_id, first_landmark.get_euclidean_representation());
    println!("Landmark state {} : {}", landmark_w.get_id().unwrap(), landmark_rel);

    // let pnp_config_cam = sfm_config_fundamental.generate_pnp_config_from_cam_id(0);
    // let (optimized_state_pnp, _) = run_pnp(&pnp_config_cam,&runtime_parameters_pnp);
    // sfm_config_fundamental.update_state(&optimized_state_pnp);
    // let cam_pos_pnp = optimized_state_pnp.get_camera_positions().first().unwrap();
    // println!("Cam 0 state pnp: {}", cam_pos_pnp);

    // let pnp_config_cam = sfm_config_fundamental.generate_pnp_config_from_cam_id(1);
    // let (optimized_state_pnp, _) = run_pnp(&pnp_config_cam,&runtime_parameters_pnp);
    // sfm_config_fundamental.update_state(&optimized_state_pnp);
    // let cam_pos_pnp = optimized_state_pnp.get_camera_positions().first().unwrap();
    // println!("Cam 0 state pnp: {}", cam_pos_pnp);

    // let (optimized_state, state_debug_list) = run_ba::<_,6,InverseLandmark<Float>,_,_>(&sfm_config_fundamental, &runtime_parameters, &trajectories);
    // sfm_config_fundamental.update_state(&optimized_state);
    // let world_cam_2 = sfm_config_fundamental.abs_pose_map().get(&1).unwrap().clone();
    // println!("Cam 2 config: {}", world_cam_2);


    let state_serialized = serde_yaml::to_string(&optimized_state.to_serial());
    let debug_states_serialized = serde_yaml::to_string(&state_debug_list);
    std::fs::write(format!("{}/ba.txt",runtime_conf.output_path), state_serialized?).expect("Unable to write file");

    Ok(())
}