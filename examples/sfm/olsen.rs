extern crate nalgebra as na;
extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;

use std::fs;
use na::Matrix4;
use vision::io::{olsen_loader::OlssenData};
use vision::sensors::camera::perspective::Perspective;
use vision::image::{features::{Match,ImageFeature}, epipolar::{compute_initial_cam_motions, EssentialDecomposition}};
use vision::sfm::{bundle_adjustment:: run_ba};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::Float;
use vision::visualize;


fn main() -> Result<()> {
    color_eyre::install()?;

    println!("--------");


    let data_ceiling_barcelona_path = "D:/Workspace/Datasets/Olsen/Ceiling_Barcelona/";
    let data_set_door_path = "D:/Workspace/Datasets/Olsen/Door_Lund/";
    let data_set_ahlströmer_path = "D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/";
    let data_set_fountain_path = "D:/Workspace/Datasets/Olsen/fountain/";
    let data_set_vasa_path = "D:/Workspace/Datasets/Olsen/vasa_statue/";
    let data_set_ninjo_path = "D:/Workspace/Datasets/Olsen/nijo/";
    let data_set_de_guerre_path = "D:/Workspace/Datasets/Olsen/de_guerre/";
    let data_set_fort_channing_path = "D:/Workspace/Datasets/Olsen/Fort_Channing_gate/";

    let olsen_data_path = data_set_fountain_path;

    let olsen_data = OlssenData::new(olsen_data_path);
    let positive_principal_distance = false;
    let feature_skip_count = 1;


    let (cam_intrinsics_0,cam_extrinsics_0) = olsen_data.get_camera_intrinsics_extrinsics(0,positive_principal_distance);
    let (cam_intrinsics_1,cam_extrinsics_1) = olsen_data.get_camera_intrinsics_extrinsics(1,positive_principal_distance);
    let (cam_intrinsics_2,cam_extrinsics_2) = olsen_data.get_camera_intrinsics_extrinsics(2,positive_principal_distance);
    let (cam_intrinsics_3,cam_extrinsics_3) = olsen_data.get_camera_intrinsics_extrinsics(3,positive_principal_distance);
    let (cam_intrinsics_4,cam_extrinsics_4) = olsen_data.get_camera_intrinsics_extrinsics(4,positive_principal_distance);
    let (cam_intrinsics_5,cam_extrinsics_5) = olsen_data.get_camera_intrinsics_extrinsics(5,positive_principal_distance);
    let (cam_intrinsics_6,cam_extrinsics_6) = olsen_data.get_camera_intrinsics_extrinsics(6,positive_principal_distance);
    let (cam_intrinsics_7,cam_extrinsics_7) = olsen_data.get_camera_intrinsics_extrinsics(7,positive_principal_distance);
    let (cam_intrinsics_8,cam_extrinsics_8) = olsen_data.get_camera_intrinsics_extrinsics(8,positive_principal_distance);
    let (cam_intrinsics_9,cam_extrinsics_9) = olsen_data.get_camera_intrinsics_extrinsics(9,positive_principal_distance);
    let (cam_intrinsics_10,cam_extrinsics_10) = olsen_data.get_camera_intrinsics_extrinsics(10,positive_principal_distance);
    let (cam_intrinsics_11,cam_extrinsics_11) = olsen_data.get_camera_intrinsics_extrinsics(11,positive_principal_distance);
    let (cam_intrinsics_12,cam_extrinsics_12) = olsen_data.get_camera_intrinsics_extrinsics(12,positive_principal_distance);
    let (cam_intrinsics_13,cam_extrinsics_13) = olsen_data.get_camera_intrinsics_extrinsics(13,positive_principal_distance);
    //let (cam_intrinsics_17,cam_extrinsics_17) = olsen_data.get_camera_intrinsics_extrinsics(17,positive_principal_distance);
    //let (cam_intrinsics_20,cam_extrinsics_20) = olsen_data.get_camera_intrinsics_extrinsics(20,positive_principal_distance);


    let matches_0_1 = olsen_data.get_matches_between_images(0, 1);
    let matches_0_2 = olsen_data.get_matches_between_images(0, 2);
    let matches_1_2 = olsen_data.get_matches_between_images(1, 2);
    let matches_2_3 = olsen_data.get_matches_between_images(2, 3);
    let matches_3_2 = olsen_data.get_matches_between_images(3, 2);
    let matches_3_4 = olsen_data.get_matches_between_images(3, 4);
    let matches_3_5 = olsen_data.get_matches_between_images(3, 5);
    let matches_4_5 = olsen_data.get_matches_between_images(4, 5);
    let matches_5_6 = olsen_data.get_matches_between_images(5, 6);
    let matches_5_7 = olsen_data.get_matches_between_images(5, 7);
    let matches_5_8 = olsen_data.get_matches_between_images(5, 8);

    let matches_6_0 = olsen_data.get_matches_between_images(6, 0);
    let matches_6_1 = olsen_data.get_matches_between_images(6, 1);
    let matches_6_2 = olsen_data.get_matches_between_images(6, 2);
    let matches_6_3 = olsen_data.get_matches_between_images(6, 3);
    let matches_6_4 = olsen_data.get_matches_between_images(6, 4);
    let matches_6_5 = olsen_data.get_matches_between_images(6, 5);
    let matches_6_7 = olsen_data.get_matches_between_images(6, 7);
    let matches_6_8 = olsen_data.get_matches_between_images(6, 8);
    let matches_6_9 = olsen_data.get_matches_between_images(6, 9);
    let matches_6_10 = olsen_data.get_matches_between_images(6, 10);
    let matches_6_11 = olsen_data.get_matches_between_images(6, 11);
    let matches_6_12 = olsen_data.get_matches_between_images(6, 12);
    let matches_6_13 = olsen_data.get_matches_between_images(6, 13);
    //let matches_6_17 = olsen_data.get_matches_between_images(6, 17);
    //let matches_17_20 = olsen_data.get_matches_between_images(17, 20);

    let matches_0_2_subvec = matches_0_2.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_1_2_subvec = matches_1_2.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_2_3_subvec = matches_2_3.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_3_2_subvec = matches_2_3.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_3_4_subvec = matches_3_4.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_3_5_subvec = matches_3_4.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_4_5_subvec = matches_4_5.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_5_6_subvec = matches_5_6.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_5_7_subvec = matches_5_7.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_5_8_subvec = matches_5_8.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();

    let matches_6_0_subvec = matches_6_0.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_1_subvec = matches_6_1.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_2_subvec = matches_6_2.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_3_subvec = matches_6_3.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_4_subvec = matches_6_4.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_5_subvec = matches_6_5.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_7_subvec = matches_6_7.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_8_subvec = matches_6_8.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_9_subvec = matches_6_9.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_10_subvec = matches_6_10.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_11_subvec = matches_6_11.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_12_subvec = matches_6_12.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_6_13_subvec = matches_6_13.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    //let matches_6_17_subvec = matches_6_17.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    //let matches_17_20_subvec = matches_6_17.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();


    // let matches_0_5 = olsen_data.get_matches_between_images(0, 5);
    // println!("matches between 0 and 5 are: #{}", matches_0_5.len());
    
    // let matches_0_10 = olsen_data.get_matches_between_images(0, 10);
    // println!("matches between 0 and 10 are: #{}", matches_0_10.len());
    
    // let matches_0_20 = olsen_data.get_matches_between_images(0, 20);
    // println!("matches between 0 and 20 are: #{}", matches_0_20.len());

    let pinhole_cam_0 = Perspective::from_matrix(&cam_intrinsics_0, false);
    let pinhole_cam_1 = Perspective::from_matrix(&cam_intrinsics_1, false);
    let pinhole_cam_2 = Perspective::from_matrix(&cam_intrinsics_2, false);
    let pinhole_cam_3 = Perspective::from_matrix(&cam_intrinsics_3, false);
    let pinhole_cam_4 = Perspective::from_matrix(&cam_intrinsics_4, false);
    let pinhole_cam_5 = Perspective::from_matrix(&cam_intrinsics_5, false);
    let pinhole_cam_6 = Perspective::from_matrix(&cam_intrinsics_6, false);
    let pinhole_cam_7 = Perspective::from_matrix(&cam_intrinsics_7, false);
    let pinhole_cam_8 = Perspective::from_matrix(&cam_intrinsics_8, false);
    let pinhole_cam_9 = Perspective::from_matrix(&cam_intrinsics_9, false);
    let pinhole_cam_10 = Perspective::from_matrix(&cam_intrinsics_10, false);
    let pinhole_cam_11 = Perspective::from_matrix(&cam_intrinsics_11, false);
    let pinhole_cam_12 = Perspective::from_matrix(&cam_intrinsics_12, false);
    let pinhole_cam_13 = Perspective::from_matrix(&cam_intrinsics_13, false);
    //let pinhole_cam_17 = Perspective::from_matrix(&cam_intrinsics_17, false);
    //let pinhole_cam_20 = Perspective::from_matrix(&cam_intrinsics_17, false);



    let mut all_matches = Vec::<Vec<Match<ImageFeature>>>::with_capacity(10);
    //all_matches.push(matches_0_1_subvec);
    //all_matches.push(matches_1_2_subvec);
    //all_matches.push(matches_2_3_subvec);
    //all_matches.push(matches_3_4_subvec);
    //all_matches.push(matches_3_2_subvec);
    //all_matches.push(matches_4_5_subvec);
    //all_matches.push(matches_3_5_subvec);
    //all_matches.push(matches_5_6_subvec);
    //all_matches.push(matches_5_7_subvec);
    //all_matches.push(matches_6_0_subvec);
    all_matches.push(matches_6_1_subvec);
    all_matches.push(matches_6_2_subvec);
    all_matches.push(matches_6_3_subvec);
    all_matches.push(matches_6_4_subvec);
    all_matches.push(matches_6_5_subvec);
    all_matches.push(matches_6_7_subvec);
    all_matches.push(matches_6_8_subvec);
    all_matches.push(matches_6_9_subvec);
    //all_matches.push(matches_6_10_subvec);
    //all_matches.push(matches_6_11_subvec);
    //all_matches.push(matches_6_12_subvec);
    //all_matches.push(matches_6_13_subvec);
    //all_matches.push(matches_6_9_subvec);
    //all_matches.push(matches_6_17_subvec);
    //all_matches.push(matches_17_20_subvec);

    for m in &all_matches {
        assert!(m.len() > 0);
    }
    
    let mut camera_data = Vec::<((usize,Perspective),(usize,Perspective))>::with_capacity(10); 
    //camera_data.push(((0,pinhole_cam_0),(1,pinhole_cam_1)));
    //camera_data.push(((1,pinhole_cam_1),(2,pinhole_cam_2)));
    //camera_data.push(((2,pinhole_cam_2),(3,pinhole_cam_3)));
    //camera_data.push(((3,pinhole_cam_3),(4,pinhole_cam_4)));
    //camera_data.push(((3,pinhole_cam_3),(2,pinhole_cam_2)));
    //camera_data.push(((4,pinhole_cam_4),(5,pinhole_cam_5)));
    //camera_data.push(((3,pinhole_cam_3),(5,pinhole_cam_5)));
    //camera_data.push(((5,pinhole_cam_5),(6,pinhole_cam_6)));
    //camera_data.push(((5,pinhole_cam_5),(7,pinhole_cam_7)));
    //camera_data.push(((6,pinhole_cam_6),(0,pinhole_cam_0)));
    camera_data.push(((6,pinhole_cam_6),(1,pinhole_cam_1)));
    camera_data.push(((6,pinhole_cam_6),(2,pinhole_cam_2)));
    camera_data.push(((6,pinhole_cam_6),(3,pinhole_cam_3)));
    camera_data.push(((6,pinhole_cam_6),(4,pinhole_cam_4)));
    camera_data.push(((6,pinhole_cam_6),(5,pinhole_cam_5)));
    camera_data.push(((6,pinhole_cam_6),(7,pinhole_cam_7)));
    camera_data.push(((6,pinhole_cam_6),(8,pinhole_cam_8)));
    camera_data.push(((6,pinhole_cam_6),(9,pinhole_cam_9)));
    //camera_data.push(((6,pinhole_cam_6),(10,pinhole_cam_10)));
    //camera_data.push(((6,pinhole_cam_6),(11,pinhole_cam_11)));
    //camera_data.push(((6,pinhole_cam_6),(12,pinhole_cam_12)));
    //camera_data.push(((6,pinhole_cam_6),(13,pinhole_cam_13)));
    //camera_data.push(((6,pinhole_cam_6),(9,pinhole_cam_9)));
    //camera_data.push(((6,pinhole_cam_6),(17,pinhole_cam_17)));
    //camera_data.push(((17,pinhole_cam_17),(20,pinhole_cam_20)));

    let mut motion_list =  Vec::<((usize,Matrix4<Float>),(usize,Matrix4<Float>))>::with_capacity(10); 
    motion_list.push(((6,cam_extrinsics_6),(4,cam_extrinsics_4)));
    motion_list.push(((6,cam_extrinsics_6),(5,cam_extrinsics_5)));
    motion_list.push(((6,cam_extrinsics_6),(7,cam_extrinsics_7)));




    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![180; 1],
        eps: vec![1e-6],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1e0],
        lm: true,
        weighting: false,
        debug: true,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::HuberWeightForPos {})
    };

    let initial_cam_poses = compute_initial_cam_motions(&all_matches, &camera_data, 1.0,-1.0, EssentialDecomposition::KANATANI);
    //let initial_cam_poses = Some(OlssenData::get_relative_motions(&motion_list));
    //let initial_cam_poses = None;

    if initial_cam_poses.is_some(){
        // for (_,(t,r)) in initial_cam_poses.as_ref().unwrap() {
        //     println!("t : {}",t);
        //     println!("r : {}",r);
        //     println!("-------");

        // }
    }

    for i in 0..camera_data.len() {
        let ((id_a,_),(id_b,_)) = camera_data[i];
        let intensity = 3.0*(olsen_data.images[id_a].buffer.max() as Float)/4.0;
        let matches_vis = visualize::display_matches_for_pyramid(&olsen_data.images[id_a],&olsen_data.images[id_b],&all_matches[i],true,intensity ,1.0);
        matches_vis.to_image().save(format!("{}match_disp_{}_{}_orb_ba.jpg",olsen_data_path,id_a,id_b)).unwrap();
    }

    let ((cam_positions,points),(s,debug_states_serialized)) = run_ba(&all_matches, &camera_data,&initial_cam_poses, olsen_data.get_image_dim(), &runtime_parameters, 1.0,-1.0);
    fs::write(format!("D:/Workspace/Rust/Vision/output/olsen.txt"), s?).expect("Unable to write file");
    if runtime_parameters.debug {
        fs::write(format!("D:/Workspace/Rust/Vision/output/olsen_debug.txt"), debug_states_serialized?).expect("Unable to write file");
    }




    Ok(())
}