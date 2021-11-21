extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use na::Vector3;
use std::fs;
use color_eyre::eyre::Result;
use vision::image::features::{Match,orb_feature::OrbFeature};
use vision::image::pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters;
use vision::image::epipolar;
use vision::sensors::camera::{pinhole::Pinhole, Camera};


fn main() -> Result<()> {

    color_eyre::install()?;


    //TODO: camera intrinsics -investigate removing badly matched feature in the 2 image set
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    let intensity_camera_2 = intensity_camera_1.clone();
    let cameras = vec!(intensity_camera_1,intensity_camera_2);


    //let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_2_images.txt").expect("Unable to read file");
    let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_ba_slow_2_ba_slow_1_images.txt").expect("Unable to read file");
    let (orb_params,matches): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string)?;

    let feature_matches = epipolar::exatct_matches(&matches[0], orb_params.pyramid_scale, false);
    let fundamental_matrix = epipolar::eight_point(&feature_matches);

    let essential_matrix = epipolar::compute_essential(&fundamental_matrix, &intensity_camera_1.get_projection(), &intensity_camera_2.get_projection());
    let normalized_matches = epipolar::filter_matches(&fundamental_matrix, &feature_matches);
    for (l,r) in &normalized_matches {
        let t = l.transpose()*fundamental_matrix*r;
        println!("{}",t);
    }
    let (h,R) = epipolar::decompose_essential(&essential_matrix,&normalized_matches);

    println!("{}",h);
    println!("{}",R);


    Ok(())
}