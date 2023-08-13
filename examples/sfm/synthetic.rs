extern crate nalgebra as na;
use color_eyre::eyre::Result;

use std::collections::{HashMap, HashSet};
use vision::{Float,load_runtime_conf};
use vision::sfm::{triangulation::Triangulation,SFMConfig, bundle_adjustment::run_ba, epipolar::tensor::BifocalType};
use vision::sensors::camera::perspective::Perspective;
use vision::image::features::{matches::Match,image_feature::ImageFeature};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use na::{Rotation3,Isometry3,Vector3,UnitQuaternion};

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    //let file_name = "camera_features_Suzanne_trans_y.yaml";
    //let file_name = "camera_features_Suzanne_trans_x.yaml";
    //let file_name = "camera_features_Suzanne_trans_z.yaml";
    //let file_name = "camera_features_Suzanne_360_60.yaml"; // Check my there are less matches than expecteed, check rotations
    //let file_name = "camera_features_sphere_trans_x.yaml";
    //let file_name = "camera_features_sphere_trans_y.yaml";
    let file_name = "camera_features_sphere_trans_z.yaml";
    //let file_name = "camera_features_sphere_360_60.yaml";  // Check my there are less matches than expecteed, check rotations

    let path = format!("{}/{}",runtime_conf.local_data_path,file_name);
    let loaded_data = models_cv::io::deserialize_feature_matches(&path);


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
    let camera_id_pairs = vec!((0,1),(1,2));

    let match_map = camera_id_pairs.iter().map(|(id1,id2)| {
        let fm_1 = feature_map.get(id1).expect("Feature map for cam id not available!");
        let fm_2 = feature_map.get(id2).expect("Feature map for cam id not available!");

        let keys_1 = fm_1.keys().into_iter().map(|&id| id).collect::<HashSet<_>>();
        let keys_2 = fm_2.keys().into_iter().map(|&id| id).collect::<HashSet<_>>();

        let shared_keys = keys_1.intersection(&keys_2).collect::<HashSet<_>>();

        let matches = shared_keys.iter().map(|key| {
            let f_1 = fm_1.get(key).expect("Invalid key!");
            let f_2 = fm_2.get(key).expect("Invalid key!");
            let cam_1_height =  camera_map.get(id1).expect("Invalid key!").get_cy()*2.0;
            let cam_2_height =  camera_map.get(id2).expect("Invalid key!").get_cy()*2.0;

            assert!(cam_1_height.fract() == 0.0);
            assert!(cam_2_height.fract() == 0.0);

            //Synthetic model data is defined with a RHS along -Z. Point are exported as-is, so we flip them here
            let image_feature_1 = ImageFeature::new(f_1.x as Float, cam_1_height - 1.0 - (f_1.y as Float), None);
            let image_feature_2 = ImageFeature::new(f_2.x as Float, cam_2_height - 1.0 - (f_2.y as Float), None);

            Match::new(image_feature_1, image_feature_2)
        }).collect::<Vec<_>>();

        ((*id1,*id2),matches)

    }).collect::<HashMap<_,_>>();

    let paths = vec![camera_id_pairs.iter().map(|&(_,c)| c).collect::<Vec<_>>()];
    let root_id = camera_id_pairs[0].0;

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

    //TODO: Check consistency of trajectory and saving!
    let pose_map_gt_option = Some(pose_map_gt);
    //let pose_map_gt_option = None;

    let sfm_config_fundamental = SFMConfig::new(root_id, &paths, pose_map_gt_option , camera_map, &match_map, 
        BifocalType::ESSENTIAL, Triangulation::LINEAR, 1.0, 2e0, 5e2, 1.0, true, true); // Investigate epipolar thresh -> more deterministic wither lower value?
    
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
        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {}),
        cg_threshold: 1e-6,
        cg_max_it: 2e3 as usize
    };
    let (_,(s,_)) = run_ba(&sfm_config_fundamental, &runtime_parameters);
    std::fs::write(format!("{}/ba.txt",runtime_conf.output_path), s?).expect("Unable to write file");

    Ok(())
}