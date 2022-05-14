extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;
extern crate nalgebra as na;

use std::{fs,path::Path};
use std::convert::TryInto;
use color_eyre::eyre::Result;
use na::{SVector,Vector3};
use vision::image::{
    Image,
    features::{Match,orb_feature::OrbFeature,Feature, ImageFeature},
    pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters,
    epipolar
};
use vision::sensors::camera::{pinhole::Pinhole, Camera};
use vision::io::octave_loader;
use vision::Float;
use vision::visualize;


fn main() -> Result<()> {
    color_eyre::install()?;

    let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/intrinsics.txt");
    let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/rotation.txt");
    let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/5_point_synthetic/translation.txt");
    let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam1_features.txt");
    let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam2_features.txt");
    let depth_positive = true;
    let invert_focal_length = false;


    // let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/intrinsics_neg.txt");
    // let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/rotation_neg.txt");
    // let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/5_point_synthetic/translation_neg.txt");
    // let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam1_features_neg.txt");
    // let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/5_point_synthetic/cam2_features_neg.txt");
    // let depth_positive = false; 
    // let invert_focal_length = false;

    // let t = SVector::<Float,3>::new(t_raw[(0,0)],t_raw[(1,0)],t_raw[(2,0)]);
    // let intensity_camera_1 = Pinhole::new(K[(0,0)],K[(1,1)],K[(0,2)],K[(1,2)], invert_focal_length);
    // let intensity_camera_2 = intensity_camera_1.clone();
    // let mut synth_matches = Vec::<Match::<ImageFeature>>::with_capacity(5);
    // for i in 0..5 {
    //     let f1 = x1h.column(i);
    //     let f2 = x2h.column(i);
    //     let feature_one = ImageFeature::new(f1[0],f1[1]);
    //     let feature_two = ImageFeature::new(f2[0],f2[1]);
    //     let m = Match::<ImageFeature>{feature_one,feature_two};
    //     synth_matches.push(m);
    // }
    // let feature_matches = epipolar::extract_matches(&synth_matches, 1.0, false); 
    // let gt = t.cross_matrix()*(&R.transpose());
    // let factor = gt[(2,2)];
    // let gt_norm = gt.map(|x| x/factor);
    // println!("------ GT -------");
    // println!("{}",gt);
    // println!("{}",gt_norm);
    // println!("{}",t_raw);
    // println!("{}",&R);
    // println!("----------------");


    let image_name_1 = "ba_slow_1";
    let image_name_2 = "ba_slow_2";
    let depth_positive = false;
    let invert_focal_length = true;        
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, invert_focal_length);
    let intensity_camera_2 = intensity_camera_1.clone();
    let orb_matches_as_string = fs::read_to_string(format!("/home/marc/Workspace/Vision/data/orb_ba_matches_{}_{}_images_5.txt",image_name_1,image_name_2)).expect("Unable to read file");
    //let orb_matches_as_string = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_ba_slow_1_ba_slow_3_images.txt").expect("Unable to read file");
    let (orb_params,matches): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string)?;
    let feature_matches = epipolar::extract_matches(&matches[0], orb_params.pyramid_scale, false); 

    let five_feature_slice : &[Match<ImageFeature>;5] = feature_matches[..5].try_into().unwrap();
    let five_point_essential_matrix = epipolar::five_point_essential(five_feature_slice,&intensity_camera_1,&intensity_camera_2,depth_positive);
    let (t_est,R_est,_) = epipolar::decompose_essential_förstner(&five_point_essential_matrix,&feature_matches,&intensity_camera_1.get_inverse_projection(),&intensity_camera_2.get_inverse_projection(), depth_positive);
    let factor = five_point_essential_matrix[(2,2)];
    let five_point_essential_matrix_norm = five_point_essential_matrix.map(|x| x/factor);
    let fundamental_matrix = epipolar::compute_fundamental(&five_point_essential_matrix,&intensity_camera_1.get_inverse_projection(), &intensity_camera_2.get_inverse_projection());
    let epipolar_lines: Vec<(Vector3<Float>, Vector3<Float>)> = five_feature_slice.iter().map(|m| epipolar::epipolar_lines(&fundamental_matrix, m)).collect();

    println!("best five point: ");
    println!("{}",five_point_essential_matrix);
    println!("{}",five_point_essential_matrix_norm);
    
    println!("----------------");
    println!("{}",t_est);
    println!("{}",R_est);

    let image_format = "png";
    let image_folder = "images";
    let image_out_folder = "output";
    let image_path_1 = format!("{}/{}.{}",image_folder,image_name_1, image_format);
    let image_path_2 = format!("{}/{}.{}",image_folder,image_name_2, image_format);
    let gray_image_1 = image_rs::open(&Path::new(&image_path_1)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();

    let mut image_1 = Image::from_gray_image(&gray_image_1, false, false, Some(image_name_1.to_string()));
    let mut image_2 = Image::from_gray_image(&gray_image_2, false, false, Some(image_name_2.to_string()));

    for m in feature_matches {
        let f1 = m.feature_one;
        let f2 = m.feature_two;

        visualize::draw_circle(&mut image_1,f1.get_x_image(), f1.get_y_image(), 5.0, 255.0);
        visualize::draw_circle(&mut image_2,f2.get_x_image(), f2.get_y_image(), 5.0, 255.0);

    }
    visualize::draw_epipolar_lines(&mut image_1, &mut image_2,255.0, &epipolar_lines);
    
    image_1.to_image().save(format!("{}/{}_epipolar_lines.{}",image_out_folder,image_name_1,image_format)).unwrap();
    image_2.to_image().save(format!("{}/{}_epipolar_lines.{}",image_out_folder,image_name_2,image_format)).unwrap();

    
    Ok(())
}