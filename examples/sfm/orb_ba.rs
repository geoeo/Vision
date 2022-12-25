extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use std::collections::HashMap;
use color_eyre::eyre::Result;
use std::path::Path;
use vision::image::pyramid::orb::{orb_runtime_parameters::OrbRuntimeParameters};
use vision::image::features::{Match,orb_feature::OrbFeature};
use vision::image::Image;
use vision::sfm::{triangulation::Triangulation, bundle_adjustment::run_ba,epipolar::tensor::BifocalType};
use vision::sensors::camera::{pinhole::Pinhole};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::sfm::SFMConfig;


fn main() -> Result<()> {

    color_eyre::install()?;

    let id_1 = "1";
    let id_2 = "2";
    let id_3 = "3";
    let id_4 = "4";

    let image_name_1 = format!("ba_slow_{}",id_1);
    let image_name_2 = format!("ba_slow_{}",id_2);
    let image_name_3 = format!("ba_slow_{}",id_3);
    let image_name_4 = format!("ba_slow_{}",id_4);


    let image_format = "png";
    let image_folder = "images/";
    let image_path_1 = format!("{}{}.{}",image_folder,image_name_1, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let image_path_3 = format!("{}{}.{}",image_folder,image_name_3, image_format);
    let image_path_4 = format!("{}{}.{}",image_folder,image_name_4, image_format);

    let gray_image_1 = image_rs::open(&Path::new(&image_path_1)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();
    let gray_image_3 = image_rs::open(&Path::new(&image_path_3)).unwrap().to_luma8();
    let gray_image_4 = image_rs::open(&Path::new(&image_path_4)).unwrap().to_luma8();

    let image_1 = Image::from_gray_image(&gray_image_1, false, false, Some(image_name_1.to_string()));
    let image_2 = Image::from_gray_image(&gray_image_2, false, false, Some(image_name_2.to_string()));
    let image_3 = Image::from_gray_image(&gray_image_3, false, false, Some(image_name_3.to_string()));
    let image_4 = Image::from_gray_image(&gray_image_4, false, false, Some(image_name_4.to_string()));

    //TODO: camera intrinsics -investigate removing badly matched feature in the 2 image set
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    let intensity_camera_2 = intensity_camera_1.clone();
    let intensity_camera_3 = intensity_camera_1.clone();
    let intensity_camera_4 = intensity_camera_1.clone();

    let orb_matches_as_string_1_2 = fs::read_to_string(format!("D:/Workspace/Rust/Vision/data/orb_ba_matches_ba_slow_{}_ba_slow_{}_images.txt",id_1,id_2)).expect("Unable to read file");
    let orb_matches_as_string_1_3 = fs::read_to_string(format!("D:/Workspace/Rust/Vision/data/orb_ba_matches_ba_slow_{}_ba_slow_{}_images.txt",id_1,id_3)).expect("Unable to read file");
    let orb_matches_as_string_1_4 = fs::read_to_string(format!("D:/Workspace/Rust/Vision/data/orb_ba_matches_ba_slow_{}_ba_slow_{}_images.txt",id_1,id_4)).expect("Unable to read file");
    let (orb_params_1_2,matches_1_2): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string_1_2)?;
    let (orb_params_1_3,matches_1_3): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string_1_3)?;
    let (orb_params_1_4,matches_1_4): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string_1_4)?;

    let mut all_matches = Vec::<Vec<Match<OrbFeature>>>::with_capacity(2);
    all_matches.extend(matches_1_2);
    //matches.extend(matches_1_3);
    //matches.extend(matches_1_4);

    let camera_map = HashMap::from([(1, intensity_camera_1), (2, intensity_camera_2)]);
    let sfm_config = SFMConfig::new(1, vec!(vec!(2)), camera_map.clone(), camera_map, vec!(all_matches), BifocalType::ESSENTIAL, Triangulation::LINEAR, true, 0.8, 1.0, true, 320*240);

    let runtime_parameters = RuntimeParameters {
        pyramid_scale: orb_params_1_2.pyramid_scale,
        max_iterations: vec![800; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e-8],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1e0],
        lm: true,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {}),
        //intensity_weighting_function:  Box::new(weighting::HuberWeightForPos {delta:1.0})
        cg_threshold: 1e-6,
        cg_max_it: 200
    };

    let (_,filtered_matches) = sfm_config.compute_lists_from_maps();

    let ((cam_positions,points),(s,debug_states_serialized)) = run_ba(&filtered_matches, &sfm_config, None, (image_1.buffer.nrows(),image_1.buffer.ncols()), &runtime_parameters, 1.0);
    fs::write(format!("D:/Workspace/Rust/Vision/output/orb_ba.txt"), s?).expect("Unable to write file");
    if runtime_parameters.debug {
        fs::write(format!("D:/Workspace/Rust/Vision/output/orb_ba_debug.txt"), debug_states_serialized?).expect("Unable to write file");
    }



    Ok(())


}