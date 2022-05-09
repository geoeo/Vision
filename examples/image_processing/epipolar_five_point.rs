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
    let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam1_features.txt");
    let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam2_features.txt");
    let depth_positive = true;

    // let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/intrinsics_neg.txt");
    // let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/rotation_neg.txt");
    // let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/5_point_synthetic/translation_neg.txt");
    // let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam1_features_neg.txt");
    // let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam2_features_neg.txt");
    // let depth_positive = false;

    let t = SVector::<Float,3>::new(t_raw[(0,0)],t_raw[(1,0)],t_raw[(2,0)]);
    let intensity_camera_1 = Pinhole::new(K[(0,0)],K[(1,1)],K[(0,2)],K[(1,2)], !depth_positive);
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
    //let orb_matches_as_string = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_ba_slow_1_ba_slow_2_images_5.txt").expect("Unable to read file");

    //let feature_matches = epipolar::extract_matches(&matches[0], orb_params.pyramid_scale, false); 


    let five_feature_slice : &[Match<ImageFeature>;5] = feature_matches[..5].try_into().unwrap();
    let five_point_essential_matrix = epipolar::five_point_essential(five_feature_slice,&intensity_camera_1,&intensity_camera_2,depth_positive);
    let factor = five_point_essential_matrix[(2,2)];
    let five_point_essential_matrix_norm = five_point_essential_matrix.map(|x| x/factor);
    println!("best five point: ");
    println!("{}",five_point_essential_matrix);
    println!("{}",five_point_essential_matrix_norm);
    
    Ok(())
}