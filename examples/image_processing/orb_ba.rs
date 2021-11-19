extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use na::{Vector3,Matrix3};
use color_eyre::eyre::Result;
use std::path::Path;
use vision::Float;
use vision::image::pyramid::orb::{orb_runtime_parameters::OrbRuntimeParameters};
use vision::image::features::{Match,orb_feature::OrbFeature, Feature};
use vision::image::Image;
use vision::image::bundle_adjustment::{camera_feature_map::CameraFeatureMap, solver::optimize};
use vision::sensors::camera::{Camera,pinhole::Pinhole};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::image::epipolar;

fn main() -> Result<()> {

    color_eyre::install()?;

    let image_name_1 = "ba_slow_1";
    let image_name_2 = "ba_slow_2";

    let image_name_3 = "ba_slow_3";
    let image_name_4 = "ba_slow_4";


    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
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



    // //TODO: recheck maximal suppression, take best corers for all windows across all pyramid levels
    // https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj2/html/agartia3/index.html
    let pyramid_scale = 1.2; // opencv default is 1.2
    let orb_runtime_params = OrbRuntimeParameters {
        pyramid_scale: pyramid_scale,
        min_image_dimensions: (20,20),
        sigma: 2.0,
        blur_radius: 5.0,
        max_features_per_octave: 8, // 15
        max_features_per_octave_scale: 1.2,
        octave_count: 8, // opencv default is 8
        harris_k: 0.04,
        harris_window_size: 5,  // 5
        fast_circle_radius: 3, //3 
        fast_threshold_factor: 0.2,
        fast_consecutive_pixels: 12, // 12
        fast_features_per_grid: 3, // 3
        fast_grid_size: (15,15),  // 15
        fast_grid_size_scale_base: 1.2,
        fast_offsets: (20,20),
        fast_offset_scale_base: 1.2,
        brief_features_to_descriptors: 128, // 128
        brief_n: 256,
        brief_s: 31,
        brief_s_scale_base: pyramid_scale,
        brief_matching_min_threshold: 256/2, //256/2
        brief_lookup_table_step: 30.0,
        brief_sampling_pattern_seed: 0x0DDB1A5ECBAD5EEDu64,
        brief_use_opencv_sampling_pattern: true
    };

    //TODO: camera intrinsics -investigate removing badly matched feature in the 2 image set
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    let intensity_camera_2 = intensity_camera_1.clone();
    // let intensity_camera_3 = intensity_camera_1.clone();
    // let intensity_camera_4 = intensity_camera_1.clone();

    let cameras = vec!(intensity_camera_1,intensity_camera_2);
    let image_pairs = vec!((&image_1, &orb_runtime_params, &image_2, &orb_runtime_params));

    // let cameras = vec!(intensity_camera_1,intensity_camera_2,intensity_camera_3,intensity_camera_4);
    // let image_pairs = vec!((&image_1, &orb_runtime_params, &image_2, &orb_runtime_params), ((&image_3, &orb_runtime_params, &image_4, &orb_runtime_params)));


    let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_ba_slow_3_ba_slow_1_images.txt").expect("Unable to read file");
    let matches: Vec<Vec<Match<OrbFeature>>> = serde_yaml::from_str(&orb_matches_read)?;


    let mut feature_map = CameraFeatureMap::new(&matches);
    feature_map.add_images_from_params(&image_1, orb_runtime_params.max_features_per_octave,orb_runtime_params.octave_count);
    feature_map.add_images_from_params(&image_2, orb_runtime_params.max_features_per_octave,orb_runtime_params.octave_count);
    // feature_map.add_images_from_params(&image_3, orb_runtime_params.max_features_per_octave,orb_runtime_params.octave_count);
    // feature_map.add_images_from_params(&image_4, orb_runtime_params.max_features_per_octave,orb_runtime_params.octave_count);

    feature_map.add_matches(&image_pairs.into_iter().map(|(i1,_,i2,_)| (i1,i2)).collect(),&matches, orb_runtime_params.pyramid_scale);
    let fundamental_matrices = matches.iter().map(|m| epipolar::eight_point(m,pyramid_scale)).collect::<Vec<epipolar::Fundamental>>();
    let essential_matrices = fundamental_matrices.iter().enumerate().map(|(i,f)| epipolar::compute_essential(f, &cameras[2*i].get_projection(), &cameras[2*i+1].get_projection())).collect::<Vec<epipolar::Essential>>();

    let normalized_matches = fundamental_matrices.iter().zip(matches.iter()).map(|(f,m)| epipolar::filter_matches(f, m, pyramid_scale)).collect::<Vec<Vec<(Vector3<Float>,Vector3<Float>)>>>();
    let initial_motion_decomp = essential_matrices.iter().enumerate().map(|(i,e)| epipolar::decompose_essential(e,&normalized_matches[i])).collect::<Vec<(Vector3<Float>,Matrix3<Float>)>>();

    let mut state = feature_map.get_initial_state(&initial_motion_decomp);
    let observed_features = feature_map.get_observed_features();
    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![50; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e-8],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1e0],
        lm: true,
        weighting: true,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: true }), 
        intensity_weighting_function:  Box::new(weighting::TrivialWeight {})
    };

    let (before_cam_positions,before_points) = state.lift();

    optimize(&mut state, &cameras, &observed_features, &runtime_parameters);

    let (cam_positions,points) = state.lift();

    println!("Cam Positions");
    for cam_pos in cam_positions {
        println!("{}",cam_pos);
    }

    println!("Points");
    for point in points {
        println!("{}",point);
    }


    //TODO: make this work with images of different sizes
    println!("{}",matches.len());



    Ok(())


}