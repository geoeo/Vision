extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use na::{Vector2,Vector3,Matrix3};
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

    let id_1 = "1";
    let id_2 = "2";

    let image_name_1 = format!("ba_slow_{}",id_1);
    let image_name_2 = format!("ba_slow_{}",id_2);


    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path_1 = format!("{}{}.{}",image_folder,image_name_1, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);

    let gray_image_1 = image_rs::open(&Path::new(&image_path_1)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();

    let image_1 = Image::from_gray_image(&gray_image_1, false, false, Some(image_name_1.to_string()));
    let image_2 = Image::from_gray_image(&gray_image_2, false, false, Some(image_name_2.to_string()));

    //TODO: camera intrinsics -investigate removing badly matched feature in the 2 image set
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    let intensity_camera_2 = intensity_camera_1.clone();
    let intensity_camera_3 = intensity_camera_1.clone();
    // let intensity_camera_4 = intensity_camera_1.clone();

    let cameras = vec!(intensity_camera_1,intensity_camera_2);
    let image_pairs = vec!((&image_1, &image_2));

    // let cameras = vec!(intensity_camera_1,intensity_camera_2,intensity_camera_3,intensity_camera_4);
    // let image_pairs = vec!((&image_1, &orb_runtime_params, &image_2, &orb_runtime_params), ((&image_3, &orb_runtime_params, &image_4, &orb_runtime_params)));


    let orb_matches_as_string = fs::read_to_string(format!("D:/Workspace/Rust/Vision/output/orb_ba_matches_ba_slow_{}_ba_slow_{}_images.txt",id_1,id_2)).expect("Unable to read file");
    let (orb_params,matches): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string)?;


    let mut feature_map = CameraFeatureMap::new(&matches);
    feature_map.add_images_from_params(&image_1, orb_params.max_features_per_octave,orb_params.octave_count);
    feature_map.add_images_from_params(&image_2, orb_params.max_features_per_octave,orb_params.octave_count);

    feature_map.add_matches(&image_pairs,&matches, orb_params.pyramid_scale);

    let feature_machtes = matches.iter().map(|m| epipolar::exatct_matches(m, orb_params.pyramid_scale, false)).collect::<Vec<Vec<(Vector2<Float>,Vector2<Float>)>>>();
    let fundamental_matrices = feature_machtes.iter().map(|m| epipolar::eight_point(m)).collect::<Vec<epipolar::Fundamental>>();
    let essential_matrices = fundamental_matrices.iter().enumerate().map(|(i,f)| epipolar::compute_essential(f, &cameras[2*i].get_projection(), &cameras[2*i+1].get_projection())).collect::<Vec<epipolar::Essential>>();

    let normalized_matches = fundamental_matrices.iter().zip(feature_machtes.iter()).map(|(f,m)| epipolar::filter_matches(f, m)).collect::<Vec<Vec<(Vector3<Float>,Vector3<Float>)>>>();
    let initial_motion_decomp = essential_matrices.iter().enumerate().map(|(i,e)| epipolar::decompose_essential(e,&normalized_matches[i])).collect::<Vec<(Vector3<Float>,Matrix3<Float>)>>();

    let mut state = feature_map.get_initial_state(&initial_motion_decomp);
    let observed_features = feature_map.get_observed_features();
    let runtime_parameters = RuntimeParameters {
        pyramid_scale: orb_params.pyramid_scale,
        max_iterations: vec![15; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e-8],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1e0],
        lm: true,
        weighting: false,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: true }), 
        intensity_weighting_function:  Box::new(weighting::TrivialWeight {})
    };

    optimize(&mut state, &cameras, &observed_features, &runtime_parameters);

    //TODO: make this a serializable datatype
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


    let s = serde_yaml::to_string(&state.to_serial())?;
    fs::write(format!("D:/Workspace/Rust/Vision/output/orb_ba_{}_{}_images.txt",image_name_1,image_name_2), s).expect("Unable to write file");


    Ok(())


}