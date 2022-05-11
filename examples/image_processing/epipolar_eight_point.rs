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


    let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_intrinsics.txt");
    let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_rotation.txt");
    let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_translation.txt");
    let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_cam1_features.txt");
    let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_cam2_features.txt");
    let depth_positive = true;
    let invert_focal_length = false;

    // let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/intrinsics_neg.txt");
    // let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/rotation_neg.txt");
    // let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/8_point_synthetic/translation_neg.txt");
    // let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/cam1_features_neg.txt");
    // let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/cam2_features_neg.txt");
    // let depth_positive = false;
    // let invert_focal_length = false;

    let t = SVector::<Float,3>::new(t_raw[(0,0)],t_raw[(1,0)],t_raw[(2,0)]);
    let intensity_camera_1 = Pinhole::new(K[(0,0)],K[(1,1)],K[(0,2)],K[(1,2)], invert_focal_length);
    let intensity_camera_2 = intensity_camera_1.clone();
    let mut synth_matches = Vec::<Match::<ImageFeature>>::with_capacity(x1h.ncols());
    for i in 0..x1h.ncols() {
        let f1 = x1h.column(i);
        let f2 = x2h.column(i);
        let feature_one = ImageFeature::new(f1[0],f1[1]);
        let feature_two = ImageFeature::new(f2[0],f2[1]);
        let m = Match::<ImageFeature>{feature_one,feature_two};
        synth_matches.push(m);
    }
    let feature_matches = epipolar::extract_matches(&synth_matches, 1.0, false); 
    let gt = (intensity_camera_1.get_projection().transpose())*t.cross_matrix()*(&R.transpose())*intensity_camera_2.get_projection();
    let factor = gt[(2,2)];
    let gt_norm = gt.map(|x| x/factor);
    println!("------ GT -------");
    println!("{}",gt_norm);
    println!("----------------");


    // let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    // let intensity_camera_2 = intensity_camera_1.clone();
    // //let orb_matches_read = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_2_images.txt").expect("Unable to read file");
    // let orb_matches_as_string = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_ba_slow_1_ba_slow_3_images.txt").expect("Unable to read file");
    // //let orb_matches_as_string = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_ba_slow_3_ba_slow_1_images.txt").expect("Unable to read file");
    // let (orb_params,matches): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string)?;
    // let feature_matches = epipolar::extract_matches(&matches[0], orb_params.pyramid_scale, false); 


    let fundamental_matrix = epipolar::eight_point(&feature_matches).transpose();
    let factor = fundamental_matrix[(2,2)];
    let fundamental_matrix_norm = fundamental_matrix.map(|x| x/factor);
    println!("best 8 point: ");
    println!("{}",fundamental_matrix);
    println!("{}",fundamental_matrix_norm);

    let essential_matrix = epipolar::compute_essential(&fundamental_matrix, &intensity_camera_1.get_inverse_projection(), &intensity_camera_2.get_inverse_projection());
    let normalized_matches = epipolar::filter_matches_from_fundamental(&fundamental_matrix, &feature_matches,1.0); 
    for m in &normalized_matches {
        let start =m.feature_one.get_as_2d_homogeneous();
        let finish = m.feature_two.get_as_2d_homogeneous();
        let t = start.transpose()*fundamental_matrix*finish;
        println!("{}",t);
    }
    //let (h,R) = epipolar::decompose_essential_kanatani(&essential_matrix,&normalized_matches, false);
    let (h,R_est, e_corrected) = epipolar::decompose_essential_förstner(&essential_matrix,&normalized_matches,&intensity_camera_1.get_inverse_projection(),&intensity_camera_2.get_inverse_projection(),depth_positive);

    println!("{}",h);
    println!("{}",t_raw);
    println!("-------");
    println!("{}",R_est);
    println!("{}",&R);


    Ok(())
}