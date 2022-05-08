extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::fs;
use std::convert::TryInto;
use color_eyre::eyre::Result;
use na::SVector;
use vision::image::features::{Match,orb_feature::OrbFeature,Feature, ImageFeature};
use vision::image::pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters;
use vision::image::epipolar;
use vision::sensors::camera::{pinhole::Pinhole, Camera};
use vision::io::octave_loader;
use vision::Float;


fn main() -> Result<()> {

    color_eyre::install()?;


    let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/intrinsics.txt");
    let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/rotation.txt");
    let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/5_point_synthetic/translation.txt");
    let t = SVector::<Float,3>::new(t_raw[(0,0)],t_raw[(1,0)],t_raw[(2,0)]);
    let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam1_features.txt");
    let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam2_features.txt");

    let intensity_camera_1 = Pinhole::new(K[(0,0)],K[(1,1)],K[(0,2)],K[(1,2)], false);
    let intensity_camera_2 = intensity_camera_1.clone();
    let mut synth_matches = Vec::<Match::<ImageFeature>>::with_capacity(5);
    for i in 0..5 {
        let f1 = x1h.column(i);
        let f2 = x2h.column(i);
        let feature_one = ImageFeature::new(f1[0],f1[1]);
        let feature_two = ImageFeature::new(f2[0],f2[1]);
        let m = Match::<ImageFeature>{feature_one,feature_two};
        synth_matches.push(m);
    }
    let feature_matches = epipolar::extract_matches(&synth_matches, 1.0, false); 
    let gt = t.cross_matrix()*R;
    let factor = gt[(2,2)];
    let gt_norm = gt.map(|x| x/factor);
    println!("------ GT -------");
    println!("{}",gt_norm);
    println!("----------------");



    //let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    //let intensity_camera_2 = intensity_camera_1.clone();
    //let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/data/orb_ba_matches_2_images.txt").expect("Unable to read file");
    let orb_matches_as_string = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_ba_slow_1_ba_slow_2_images_5.txt").expect("Unable to read file");
    //let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/data/orb_ba_matches_ba_slow_1_ba_slow_3_images.txt").expect("Unable to read file");
    //let orb_matches_as_string = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_ba_slow_1_ba_slow_3_images.txt").expect("Unable to read file");
    //let (orb_params,matches): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string)?;
    //let feature_matches = epipolar::extract_matches(&matches[0], orb_params.pyramid_scale, false); 


    let five_feature_slice : &[Match<ImageFeature>;5] = feature_matches[..5].try_into().unwrap();
    let five_point_essential_matrix = epipolar::five_point_essential(five_feature_slice,&intensity_camera_1,&intensity_camera_2,true);
    println!("best five point: ");
    println!("{}",five_point_essential_matrix);
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
    let (h,R, e_corrected) = epipolar::decompose_essential_förstner(&essential_matrix,&normalized_matches, false);

    println!("{}",h);
    println!("{}",R);
    println!("{}", R*h);


    Ok(())
}