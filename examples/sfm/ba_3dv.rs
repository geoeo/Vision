extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use color_eyre::eyre::Result;
use vision::image::features::{Match,ImageFeature};
use vision::sfm::bundle_adjustment::run_ba;
use vision::sensors::camera::pinhole::Pinhole;
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::io::three_dv_loader;

fn main() -> Result<()> {

    color_eyre::install()?;

    let matches_0_1 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_1.xyz");
    let matches_0_2 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_2.xyz");
    let matches_1_3 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation_neg_z_no_noise_1.xyz", "image_formation_neg_z_no_noise_3.xyz");

    // let matches_0_1 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation0_neg_z.xyz", "image_formation1_neg_z.xyz");
    // let matches_0_2 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation0_neg_z.xyz", "image_formation2_neg_z.xyz");
    // let matches_1_3 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation1_neg_z.xyz", "image_formation3_neg_z.xyz");
    let intensity_camera_0 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_1 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_2 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_3 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);

    let mut all_matches = Vec::<Vec<Match<ImageFeature>>>::with_capacity(2);
    all_matches.push(matches_0_1);
    all_matches.push(matches_0_2);
    all_matches.push(matches_1_3);

    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![300; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1e0],
        lm: true,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {})
    };


    let camera_data = vec!(((0,intensity_camera_0),(1,intensity_camera_1)),((0,intensity_camera_0),(2,intensity_camera_2)),((1,intensity_camera_0),(3,intensity_camera_3)));

    let ((cam_positions,points),(s,debug_states_serialized)) = run_ba(&all_matches, &camera_data,&None, (480,640), &runtime_parameters, 1.0,-1.0);
    fs::write(format!("D:/Workspace/Rust/Vision/output/3dv.txt"), s?).expect("Unable to write file");
    if runtime_parameters.debug {
        fs::write(format!("D:/Workspace/Rust/Vision/output/3dv_debug.txt"), debug_states_serialized?).expect("Unable to write file");
    }
   

    Ok(())


}