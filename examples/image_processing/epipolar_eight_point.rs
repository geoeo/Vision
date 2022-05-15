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
use vision::sensors::camera::{pinhole::Pinhole, Camera, perspective::Perspective};
use vision::io::{octave_loader,olsen_loader::OlssenData};
use vision::Float;
use vision::visualize;
use vision::numerics::pose;

fn main() -> Result<()> {

    color_eyre::install()?;


    // let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_intrinsics.txt");
    // let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_rotation.txt");
    // let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_translation.txt");
    // let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_cam1_features.txt");
    // let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/eight_point_cam2_features.txt");
    // let depth_positive = true;
    // let invert_focal_length = false;

    // let K = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/intrinsics_neg.txt");
    // let R = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/rotation_neg.txt");
    // let t_raw = octave_loader::load_vector("/home/marc/Workspace/Vision/data/8_point_synthetic/translation_neg.txt");
    // let x1h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/cam1_features_neg.txt");
    // let x2h = octave_loader::load_matrix("/home/marc/Workspace/Vision/data/8_point_synthetic/cam2_features_neg.txt");
    // let depth_positive = false;
    // let invert_focal_length = false;

    // let t = SVector::<Float,3>::new(t_raw[(0,0)],t_raw[(1,0)],t_raw[(2,0)]);
    // let intensity_camera_1 = Pinhole::new(K[(0,0)],K[(1,1)],K[(0,2)],K[(1,2)], invert_focal_length);
    // let intensity_camera_2 = intensity_camera_1.clone();
    // let mut synth_matches = Vec::<Match::<ImageFeature>>::with_capacity(x1h.ncols());
    // for i in 0..x1h.ncols() {
    //     let f1 = x1h.column(i);
    //     let f2 = x2h.column(i);
    //     let feature_one = ImageFeature::new(f1[0],f1[1]);
    //     let feature_two = ImageFeature::new(f2[0],f2[1]);
    //     let m = Match::<ImageFeature>{feature_one,feature_two};
    //     synth_matches.push(m);
    // }
    // let feature_matches = epipolar::extract_matches(&synth_matches, 1.0, false); 
    // let gt = (intensity_camera_1.get_projection().transpose())*t.cross_matrix()*(&R.transpose())*intensity_camera_2.get_projection();
    // let factor = gt[(2,2)];
    // let gt_norm = gt.map(|x| x/factor);
    // println!("------ GT -------");
    // println!("{}",gt_norm);
    // println!("{}",t_raw);
    // println!("{}",&R);
    // println!("----------------");


    // let image_name_1 = "ba_slow_3";
    // let image_name_2 = "ba_slow_1";
    // let image_format = "png";
    // let image_folder = "images";
    // let image_path_1 = format!("{}/{}.{}",image_folder,image_name_1, image_format);
    // let image_path_2 = format!("{}/{}.{}",image_folder,image_name_2, image_format);
    // let depth_positive = false;
    // let invert_focal_length = true;
    // let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, invert_focal_length);
    // let intensity_camera_2 = intensity_camera_1.clone();
    // //let orb_matches_read = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_2_images.txt").expect("Unable to read file");
    // //let orb_matches_as_string = fs::read_to_string("/home/marc/Workspace/Vision/data/orb_ba_matches_ba_slow_1_ba_slow_3_images.txt").expect("Unable to read file");
    // let orb_matches_as_string = fs::read_to_string(format!("/home/marc/Workspace/Vision/data/orb_ba_matches_{}_{}_images.txt",image_name_1,image_name_2)).expect("Unable to read file");
    // let (orb_params,matches): (OrbRuntimeParameters,Vec<Vec<Match<OrbFeature>>>) = serde_yaml::from_str(&orb_matches_as_string)?;
    // let feature_matches = epipolar::extract_matches(&matches[0], orb_params.pyramid_scale, false); 

    let image_name_1 = "DSC_0001";
    let image_name_2 = "DSC_0002";
    let image_format = "jpg";
    let data_set_door_path = "/mnt/d/Workspace/Datasets/Olsen/Door_Lund/";
    let image_path_1 = format!("{}/images/{}.{}",data_set_door_path,image_name_1, image_format);
    let image_path_2 = format!("{}/images/{}.{}",data_set_door_path,image_name_2, image_format);
    let olsen_data_path = data_set_door_path;
    let epipolar_thresh = 0.5;

    let olsen_data = OlssenData::new(olsen_data_path);
    let depth_positive = false;
    let feature_skip_count = 1;

    let (cam_intrinsics_0,cam_extrinsics_0) = olsen_data.get_camera_intrinsics_extrinsics(0,depth_positive);
    let (cam_intrinsics_1,cam_extrinsics_1) = olsen_data.get_camera_intrinsics_extrinsics(1,depth_positive);
    let feature_matches = olsen_data.get_matches_between_images(0, 1);
    let intensity_camera_1 = Perspective::from_matrix(&cam_intrinsics_0, true);
    let intensity_camera_2 = Perspective::from_matrix(&cam_intrinsics_1, true);
    let p0 = pose::from_matrix(&cam_extrinsics_0);
    let p1 = pose::from_matrix(&cam_extrinsics_1);
    let p01 = pose::pose_difference(&p0, &p1);
    let (t_raw, R) = pose::decomp(&p01);
    let gt = t_raw.cross_matrix()*(&R.transpose());
    let factor = gt[(2,2)];
    let gt_norm = gt.map(|x| x/factor);
    println!("------ GT -------");
    println!("{}",gt);
    println!("{}",gt_norm);
    println!("{}",t_raw);
    println!("{}",&R);
    println!("----------------");


    let fundamental_matrix = epipolar::eight_point(&feature_matches);
    let factor = fundamental_matrix[(2,2)];
    let fundamental_matrix_norm = fundamental_matrix.map(|x| x/factor);
    println!("best 8 point: ");
    println!("{}",fundamental_matrix);
    println!("{}",fundamental_matrix_norm);

    let normalized_matches = epipolar::filter_matches_from_fundamental(&fundamental_matrix, &feature_matches,1.0); 
    for m in &normalized_matches {
        let start =m.feature_one.get_as_2d_homogeneous();
        let finish = m.feature_two.get_as_2d_homogeneous();
        let t = start.transpose()*fundamental_matrix*finish;
        //println!("{}",t);
    }
    let essential_matrix = epipolar::compute_essential(&fundamental_matrix, &intensity_camera_1.get_inverse_projection(), &intensity_camera_2.get_inverse_projection());
    //let (h,R) = epipolar::decompose_essential_kanatani(&essential_matrix,&normalized_matches, false);
    let (h,R_est, e_corrected) = epipolar::decompose_essential_förstner(&essential_matrix,&feature_matches,&intensity_camera_1.get_inverse_projection(),&intensity_camera_2.get_inverse_projection(),depth_positive);
    //let feature_matches_vis = &feature_matches[0..20];
    let feature_matches_vis = epipolar::filter_matches_from_fundamental(&fundamental_matrix, &feature_matches,0.0001); 
    let epipolar_lines: Vec<(Vector3<Float>, Vector3<Float>)> = feature_matches_vis.iter().map(|m| epipolar::epipolar_lines(&fundamental_matrix_norm, m)).collect();

    println!("{}",h);
    println!("-------");
    println!("{}",R_est);


    let image_out_folder = "output";
    let gray_image_1 = image_rs::open(&Path::new(&image_path_1)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();

    let mut image_1 = Image::from_gray_image(&gray_image_1, false, false, Some(image_name_1.to_string()));
    let mut image_2 = Image::from_gray_image(&gray_image_2, false, false, Some(image_name_2.to_string()));

    for m in feature_matches_vis.iter() {
        let f1 = &m.feature_one;
        let f2 = &m.feature_two;

        visualize::draw_circle(&mut image_1,f1.get_x_image(), f1.get_y_image(), 5.0, 255.0);
        visualize::draw_circle(&mut image_2,f2.get_x_image(), f2.get_y_image(), 5.0, 255.0);

    }
    visualize::draw_epipolar_lines(&mut image_1, &mut image_2,25.0, &epipolar_lines);
    
    image_1.to_image().save(format!("{}/{}_epipolar_lines_8p.{}",image_out_folder,image_name_1,image_format)).unwrap();
    image_2.to_image().save(format!("{}/{}_epipolar_lines_8p.{}",image_out_folder,image_name_2,image_format)).unwrap();




    Ok(())
}