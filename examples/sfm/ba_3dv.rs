extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use std::collections::HashMap;
use color_eyre::eyre::Result;
use vision::image::{features::{Match,ImageFeature}, epipolar::{compute_pairwise_cam_motions_with_filtered_matches, compute_pairwise_cam_motions_with_filtered_matches_for_path, BifocalType,EssentialDecomposition}};
use vision::sfm::bundle_adjustment::run_ba;
use vision::sensors::camera::pinhole::Pinhole;
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::io::three_dv_loader;
use vision::sfm::SFMConfig;
use vision::{float,Float,load_runtime_conf};


fn main() -> Result<()> {

    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();


    let matches_0_1 = three_dv_loader::load_matches(&format!("{}/3dv",runtime_conf.dataset_path), "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_1.xyz");
    let matches_0_2 = three_dv_loader::load_matches(&format!("{}/3dv",runtime_conf.dataset_path), "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_2.xyz");
    let matches_1_3 = three_dv_loader::load_matches(&format!("{}/3dv",runtime_conf.dataset_path), "image_formation_neg_z_no_noise_1.xyz", "image_formation_neg_z_no_noise_3.xyz");
    let matches_0_3 = three_dv_loader::load_matches(&format!("{}/3dv",runtime_conf.dataset_path), "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_3.xyz");
    let matches_2_1 = three_dv_loader::load_matches(&format!("{}/3dv",runtime_conf.dataset_path), "image_formation_neg_z_no_noise_2.xyz", "image_formation_neg_z_no_noise_1.xyz");
    let matches_2_3 = three_dv_loader::load_matches(&format!("{}/3dv",runtime_conf.dataset_path), "image_formation_neg_z_no_noise_2.xyz", "image_formation_neg_z_no_noise_3.xyz");

    // let matches_0_1 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation0_neg_z.xyz", "image_formation1_neg_z.xyz");
    // let matches_0_2 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation0_neg_z.xyz", "image_formation2_neg_z.xyz");
    // let matches_1_3 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation1_neg_z.xyz", "image_formation3_neg_z.xyz");
    let intensity_camera_0 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_1 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_2 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_3 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);


    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![300; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1e-3],
        lm: true,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {})
    };

    let camera_map = HashMap::from([(0, intensity_camera_0), (1, intensity_camera_1),(2,intensity_camera_2),(3,intensity_camera_3) ]);
    //let sfm_config = SFMConfig::new(0, vec!(vec!(2), vec!(1,3), vec!(3)), camera_map, vec!(vec!(matches_0_2),vec!(matches_0_1, matches_1_3),vec!(matches_0_3)));
    let sfm_config = SFMConfig::new(2, vec!(vec!(1), vec!(3)), camera_map, vec!(vec!(matches_2_1),vec!(matches_2_3)));

    let depth_prior = -1.0;
    let epipolar_thresh = 0.01;
    let positive_principal_distance = false;
    let normalize_features = false;

    let (initial_cam_motions_per_path,filtered_matches_per_path) = compute_pairwise_cam_motions_with_filtered_matches(
        &sfm_config,
        1.0,
        epipolar_thresh,
        positive_principal_distance,
        normalize_features,
        BifocalType::FUNDAMENTAL,
        EssentialDecomposition::FÖRSNTER
);


    let ((cam_positions,points),(s,debug_states_serialized)) = run_ba(&filtered_matches_per_path, &sfm_config, Some(&initial_cam_motions_per_path), (480,640), &runtime_parameters, 1.0,depth_prior);
    //let ((cam_positions,points),(s,debug_states_serialized)) = run_ba(&sfm_config.matches(), &sfm_config, None, (480,640), &runtime_parameters, 1.0,depth_prior);
    fs::write(format!("{}/{}",runtime_conf.local_data_path,"3dv.txt"), s?).expect("Unable to write file");
    if runtime_parameters.debug {
        fs::write(format!("{}/{}",runtime_conf.local_data_path,"3dv_debug.txt"), debug_states_serialized?).expect("Unable to write file");
    }
   

    Ok(())


}