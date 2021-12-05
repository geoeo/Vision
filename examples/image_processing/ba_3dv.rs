extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use na::{Vector3,Matrix3};
use color_eyre::eyre::Result;
use vision::Float;
use vision::image::features::{Match,ImageFeature};
use vision::image::bundle_adjustment::{camera_feature_map::CameraFeatureMap, solver::optimize};
use vision::sensors::camera::{Camera,pinhole::Pinhole};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::image::epipolar;
use vision::io::three_dv_loader;

fn main() -> Result<()> {

    color_eyre::install()?;

    //let matches = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation0.xyz", "image_formation1.xyz");
    //let matches = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_1.xyz");
    //let matches_2_3 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_3.xyz");
    let matches = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_1.xyz");
    let matches_2_3 = three_dv_loader::load_matches("D:/Workspace/Cpp/3dv_tutorial/bin/data", "image_formation_neg_z_no_noise_0.xyz", "image_formation_neg_z_no_noise_2.xyz");
    let number_of_matches = matches.len();
    let intensity_camera_0 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_1 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_2 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);
    let intensity_camera_3 = Pinhole::new(1000.0, 1000.0, 320.0, 240.0, true);

    //let cameras = vec!(intensity_camera_0,intensity_camera_1);
    let cameras = vec!(intensity_camera_0,intensity_camera_1,intensity_camera_2,intensity_camera_3);

    let mut all_matches = Vec::<Vec<Match<ImageFeature>>>::with_capacity(2);
    all_matches.push(matches);

    let mut feature_map = CameraFeatureMap::new(&all_matches,3, (480,640));
    all_matches.push(matches_2_3);
    let image_id_pairs = vec!((0,1),(0,3));
    feature_map.add_camera(vec!(0,1,3), number_of_matches,1);
    feature_map.add_matches(&image_id_pairs,&all_matches, 1.0);
    let initial_motions = vec!((Vector3::<Float>::new(0.0,0.0,0.0),Matrix3::<Float>::identity()),(Vector3::<Float>::new(0.0,0.0,0.0),Matrix3::<Float>::identity()));

    // let mut feature_map = CameraFeatureMap::new(&all_matches,2, (480,640));
    // let image_id_pairs = vec!((0,1));
    // feature_map.add_camera(vec!(0,1), number_of_matches,1);
    // feature_map.add_matches(&image_id_pairs,&all_matches, 1.0);
    // let initial_motions = vec!((Vector3::<Float>::new(0.0,0.0,0.0),Matrix3::<Float>::identity()));


    let mut state = feature_map.get_initial_state(None, -5.5);
    let observed_features = feature_map.get_observed_features();


    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![200; 1],
        eps: vec![1e0],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-15, 
        delta_eps: 1e-15,
        taus: vec![1e-3],
        lm: true,
        weighting: true,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::TrivialWeight {})
    };

    optimize(&mut state, &cameras, &observed_features, &runtime_parameters);

    //TODO: make this a serializable datatype
    let (cam_positions,points) = state.as_matrix_point();

    println!("Cam Positions");
    for cam_pos in cam_positions {
        let cam_pos_world = vision::numerics::pose::invert_se3(&cam_pos);
        println!("{}",cam_pos_world);
    }

    // println!("Points");
    // for point in points {
    //     println!("{}",point);
    // }


    let s = serde_yaml::to_string(&state.to_serial())?;
    fs::write(format!("D:/Workspace/Rust/Vision/output/3dv.txt"), s).expect("Unable to write file");
   

    Ok(())


}