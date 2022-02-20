extern crate nalgebra as na;
extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;
use na::{Vector3,Vector2,Matrix3};

use std::fs;
use vision::io::{olsen_loader::OlsenData};
use vision::sensors::camera::{Camera,pinhole::Pinhole};
use vision::image::{epipolar,features::{Match,ImageFeature}};
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
    let data_set_fountain_path = "D:/Workspace/Datasets/Olsen/fountain/";
    let data_set_vasa_path = "D:/Workspace/Datasets/Olsen/vasa_statue/";
    let data_set_ninjo_path = "D:/Workspace/Datasets/Olsen/nijo/"; // fixes spaces in names
    let data_set_de_guerre_path = "D:/Workspace/Datasets/Olsen/de_guerre/";
    let data_set_fort_channing_path = "D:/Workspace/Datasets/Olsen/Fort_Channing_gate/";

    let olsen_data = OlsenData::new(data_set_fountain_path);
    let positive_principal_distance = false;


    let (cam_intrinsics_0,cam_extrinsics_0) = olsen_data.get_camera_intrinsics_extrinsics(0,positive_principal_distance);
    let (cam_intrinsics_1,cam_extrinsics_1) = olsen_data.get_camera_intrinsics_extrinsics(1,positive_principal_distance);
    let (cam_intrinsics_2,cam_extrinsics_2) = olsen_data.get_camera_intrinsics_extrinsics(2,positive_principal_distance);
    let (cam_intrinsics_3,cam_extrinsics_3) = olsen_data.get_camera_intrinsics_extrinsics(3,positive_principal_distance);
    let (cam_intrinsics_6,cam_extrinsics_6) = olsen_data.get_camera_intrinsics_extrinsics(6,positive_principal_distance);
    let (cam_intrinsics_7,cam_extrinsics_7) = olsen_data.get_camera_intrinsics_extrinsics(7,positive_principal_distance);
    let (cam_intrinsics_8,cam_extrinsics_8) = olsen_data.get_camera_intrinsics_extrinsics(8,positive_principal_distance);

    // println!("{}",cam_intrinsics_0);
    // println!("{}",cam_extrinsics_0);

    // let pose_0 = pose::from_matrix(&cam_extrinsics_0);
    // let pose_1 = pose::from_matrix(&cam_extrinsics_1);

    // println!("{}",pose_0);
    // println!("{}",pose_1);

    // let pose_0_to_1 = pose::pose_difference(&pose_0, &pose_1);
    // println!("{}",pose_0_to_1);


    let matches_0_1 = olsen_data.get_matches_between_images(0, 1);
    let matches_0_2 = olsen_data.get_matches_between_images(0, 2);
    let matches_1_2 = olsen_data.get_matches_between_images(1, 2);
    let matches_0_3 = olsen_data.get_matches_between_images(0, 3);

    let matches_6_7 = olsen_data.get_matches_between_images(6, 7);
    let matches_7_8 = olsen_data.get_matches_between_images(7, 8);


    let feature_skip_count = 20;

    println!("matches between 0 and 1 are: #{}", matches_0_1.len());
    let matches_0_1_subvec = matches_0_1.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    println!("sub set of matches between 0 and 1 are: #{}", matches_0_1_subvec.len());

    let matches_0_2_subvec = matches_0_2.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_1_2_subvec = matches_1_2.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_0_3_subvec = matches_0_3.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_7_subvec = matches_6_7.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_7_8_subvec = matches_7_8.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();


    // let matches_0_5 = olsen_data.get_matches_between_images(0, 5);
    // println!("matches between 0 and 5 are: #{}", matches_0_5.len());
    
    // let matches_0_10 = olsen_data.get_matches_between_images(0, 10);
    // println!("matches between 0 and 10 are: #{}", matches_0_10.len());
    
    // let matches_0_20 = olsen_data.get_matches_between_images(0, 20);
    // println!("matches between 0 and 20 are: #{}", matches_0_20.len());

    let pinhole_cam_0 = Pinhole::from_matrix(&cam_intrinsics_0, false);
    let pinhole_cam_1 = Pinhole::from_matrix(&cam_intrinsics_1, false);
    let pinhole_cam_2 = Pinhole::from_matrix(&cam_intrinsics_2, false);
    let pinhole_cam_3 = Pinhole::from_matrix(&cam_intrinsics_3, false);
    let pinhole_cam_6 = Pinhole::from_matrix(&cam_intrinsics_6, false);
    let pinhole_cam_7 = Pinhole::from_matrix(&cam_intrinsics_7, false);
    let pinhole_cam_8 = Pinhole::from_matrix(&cam_intrinsics_8, false);

    //let cameras = vec!(pinhole_cam_0,pinhole_cam_1,pinhole_cam_2,pinhole_cam_3);
    let cameras = vec!(pinhole_cam_6,pinhole_cam_7,pinhole_cam_8);

    let mut all_matches = Vec::<Vec<Match<ImageFeature>>>::with_capacity(2);
    all_matches.push(matches_6_7_subvec);
    all_matches.push(matches_7_8_subvec);
    //all_matches.push(matches_1_2_subvec);
    //all_matches.push(matches_0_3_subvec);
    // This depends on matches
    let image_id_pairs = vec!((6,7),(7,8));
    // has to match cameras list
    let mut feature_map = CameraFeatureMap::new(&all_matches,vec!(6,7,8), (1296,1936));
    feature_map.add_matches(&image_id_pairs,&all_matches, 1.0);

    //Slightly buggy
    // let feature_machtes = all_matches.iter().map(|m| epipolar::extract_matches(m, 1.0, false)).collect::<Vec<Vec<(Vector2<Float>,Vector2<Float>)>>>();
    // let fundamental_matrices = feature_machtes.iter().map(|m| epipolar::eight_point(m)).collect::<Vec<epipolar::Fundamental>>();
    // let essential_matrices = fundamental_matrices.iter().enumerate().map(|(i,f)| {
    //     let (id_1,id_2) = image_id_pairs[i];
    //     let (c_1, _ ) = feature_map.camera_map[&id_1];
    //     let (c_2, _ ) = feature_map.camera_map[&id_2];
    //     epipolar::compute_essential(f, &cameras[c_1].get_projection(), &cameras[c_2].get_projection())
    // }).collect::<Vec<epipolar::Essential>>();
    // let normalized_matches = fundamental_matrices.iter().zip(feature_machtes.iter()).map(|(f,m)| epipolar::filter_matches(f, m)).collect::<Vec<Vec<(Vector3<Float>,Vector3<Float>)>>>();
    // let initial_motion_decomp = essential_matrices.iter().enumerate().map(|(i,e)| epipolar::decompose_essential_förstner(e,&normalized_matches[i])).collect::<Vec<(Vector3<Float>,Matrix3<Float>)>>();
    // for (h,R) in &initial_motion_decomp {
    //     println!("{}", h);
    // }

    let mut state = feature_map.get_euclidean_landmark_state(None, Vector3::<Float>::new(0.0,0.0,-1.0));
    let observed_features = feature_map.get_observed_features(false);


    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![50; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-10, 
        delta_eps: 1e-10,
        taus: vec![1e0],
        lm: true,
        weighting: true,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::HuberWeightForPos {})
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