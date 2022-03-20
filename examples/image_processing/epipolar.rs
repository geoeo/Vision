extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use color_eyre::eyre::Result;
use vision::image::features::{Match,orb_feature::OrbFeature,Feature};
use vision::image::pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters;
use vision::image::epipolar;
use vision::sensors::camera::{pinhole::Pinhole, Camera};


fn main() -> Result<()> {

    color_eyre::install()?;


    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    let intensity_camera_2 = intensity_camera_1.clone();


    //let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_2_images.txt").expect("Unable to read file");
    let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_ba_slow_1_ba_slow_3_images.txt").expect("Unable to read file");
    let (orb_params,matches): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string)?;

    let feature_matches = epipolar::extract_matches(&matches[0], orb_params.pyramid_scale, false);
    let fundamental_matrix = epipolar::eight_point(&feature_matches);
    let essential_matrix = epipolar::compute_essential(&fundamental_matrix, &intensity_camera_1.get_projection(), &intensity_camera_2.get_projection());
    let normalized_matches = epipolar::filter_matches_from_fundamental(&fundamental_matrix, &feature_matches,1.0);
    for m in &normalized_matches {
        let start = m.feature_one.get_as_3d_point(1.0);
        let finish = m.feature_two.get_as_3d_point(1.0);
        let t = start.transpose()*fundamental_matrix*finish;
        println!("{}",t);
    }
    //let (h,R) = epipolar::decompose_essential_kanatani(&essential_matrix,&normalized_matches, false);
    let (h,R) = epipolar::decompose_essential_förstner(&essential_matrix,&normalized_matches, false);

    println!("{}",h);
    println!("{}",R);
    println!("{}", R*h);


    Ok(())
}