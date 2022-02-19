extern crate nalgebra as na;
extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;
use na::Vector3;

use std::fs;
use vision::io::{olsen_loader::OlsenData};
use vision::sensors::camera::{Camera,pinhole::Pinhole};
use vision::image::features::{Match,ImageFeature};
use vision::sfm::{bundle_adjustment::{camera_feature_map::CameraFeatureMap, solver::optimize}};
use vision::numerics::pose;
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::Float;


fn main() -> Result<()> {
    color_eyre::install()?;

    println!("--------");

    let data_set_door_path = "D:/Workspace/Datasets/Olsen/Door_Lund/";
    let data_set_ahlströmer_path = "D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/";

    let olsen_data = OlsenData::new(data_set_door_path);
    let positive_principal_distance = false;


    let (cam_intrinsics_0,cam_extrinsics_0) = olsen_data.get_camera_intrinsics_extrinsics(0,positive_principal_distance);
    let (cam_intrinsics_1,cam_extrinsics_1) = olsen_data.get_camera_intrinsics_extrinsics(1,positive_principal_distance);

    println!("{}",cam_intrinsics_0);
    println!("{}",cam_extrinsics_0);

    let pose_0 = pose::from_matrix(&cam_extrinsics_0);
    let pose_1 = pose::from_matrix(&cam_extrinsics_1);

    println!("{}",pose_0);
    println!("{}",pose_1);

    let pose_0_to_1 = pose::pose_difference(&pose_0, &pose_1);
    println!("{}",pose_0_to_1);


    let matches_0_1 = olsen_data.get_matches_between_images(0, 1);
    let matches_0_1_subvec = (matches_0_1[..200]).into_iter().cloned().collect::<Vec<Match<ImageFeature>>>();
    println!("matches between 0 and 1 are: #{}", matches_0_1.len());

    // let matches_0_5 = olsen_data.get_matches_between_images(0, 5);
    // println!("matches between 0 and 5 are: #{}", matches_0_5.len());
    
    // let matches_0_10 = olsen_data.get_matches_between_images(0, 10);
    // println!("matches between 0 and 10 are: #{}", matches_0_10.len());
    
    // let matches_0_20 = olsen_data.get_matches_between_images(0, 20);
    // println!("matches between 0 and 20 are: #{}", matches_0_20.len());

    let pinhole_cam_0 = Pinhole::from_matrix(&cam_intrinsics_0, false);
    let pinhole_cam_1 = Pinhole::from_matrix(&cam_intrinsics_1, false);
    let cameras = vec!(pinhole_cam_0,pinhole_cam_1);

    let mut all_matches = Vec::<Vec<Match<ImageFeature>>>::with_capacity(2);
    all_matches.push(matches_0_1_subvec);
    let image_id_pairs = vec!((0,1));
    let mut feature_map = CameraFeatureMap::new(&all_matches,vec!(0,1), (1296,1936));
    feature_map.add_matches(&image_id_pairs,&all_matches, 1.0);

    let mut state = feature_map.get_euclidean_landmark_state(None, Vector3::<Float>::new(0.0,0.0,-1.0));
    let observed_features = feature_map.get_observed_features(false);


    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![200; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1e0],
        lm: true,
        weighting: true,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::TrivialWeight {})
    };

    let some_debug_state_list = optimize(&mut state, &cameras, &observed_features, &runtime_parameters);

    
    let (cam_positions,points) = state.as_matrix_point();

    let s = serde_yaml::to_string(&state.to_serial())?;
    fs::write(format!("D:/Workspace/Rust/Vision/output/olsen.txt"), s).expect("Unable to write file");
    if runtime_parameters.debug {
        let debug_states_serialized = serde_yaml::to_string(&some_debug_state_list)?;
        fs::write(format!("D:/Workspace/Rust/Vision/output/olsen_debug.txt"), debug_states_serialized).expect("Unable to write file");
    }





    Ok(())
}