extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use na::Vector3;
use std::fs;
use color_eyre::eyre::Result;
use std::path::Path;
use vision::image::pyramid::orb::{orb_runtime_parameters::OrbRuntimeParameters};
use vision::image::features::{Match,orb_feature::OrbFeature, Feature};
use vision::image::Image;
use vision::image::bundle_adjustment::{camera_feature_map::CameraFeatureMap, solver::optimize};
use vision::image::epipolar;
use vision::sensors::camera::{pinhole::Pinhole, Camera};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::Float;


fn main() -> Result<()> {

    color_eyre::install()?;

    let pyramid_scale = 1.2;
    //TODO: camera intrinsics -investigate removing badly matched feature in the 2 image set
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    let intensity_camera_2 = intensity_camera_1.clone();
    let cameras = vec!(intensity_camera_1,intensity_camera_2);


    //let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_2_images.txt").expect("Unable to read file");
    let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_ba_slow_3_ba_slow_1_images.txt").expect("Unable to read file");
    let matches: Vec<Vec<Match<OrbFeature>>> = serde_yaml::from_str(&orb_matches_read)?;

    let fundamental_matrix = epipolar::eight_point(&matches[0], pyramid_scale);

    for m in &matches[0] {
        let (x_left, y_left) = m.feature_one.1.reconstruct_original_coordiantes_for_float(pyramid_scale);
        let (x_right, y_right) = m.feature_two.1.reconstruct_original_coordiantes_for_float(pyramid_scale);

        let feature_left = Vector3::new(x_left,y_left,1.0);
        let feature_right = Vector3::new(x_right,y_right,1.0);

        let t = feature_right.transpose()*fundamental_matrix*feature_left;

        println!("{}",t);

    }

    let essential_matrix = epipolar::compute_essential(&fundamental_matrix, &intensity_camera_1.get_projection(), &intensity_camera_2.get_projection());
    let normalized_matches = epipolar::filter_matches(&fundamental_matrix, &matches[0], pyramid_scale);
    let (h,R) = epipolar::decompose_essential(&essential_matrix,&normalized_matches);

    println!("{}",h);
    println!("{}",R);


    Ok(())
}