extern crate nalgebra as na;
extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;

use std::fs;
use std::collections::HashMap;
use na::{Vector3,Matrix3,Matrix4};
use vision::io::olsen_loader::OlssenData;
use vision::sensors::camera::perspective::Perspective;
use vision::image::{features::{Match,ImageFeature}, epipolar::{compute_pairwise_cam_motions_with_filtered_matches, compute_pairwise_cam_motions_with_filtered_matches_for_path, BifocalType,EssentialDecomposition}};
use vision::sfm::{SFMConfig,bundle_adjustment::run_ba};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::{float,Float,load_runtime_conf};



fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    println!("--------");

    let data_ceiling_barcelona_path = format!("{}/Olsen/Ceiling_Barcelona/",runtime_conf.dataset_path);
    let data_set_door_path = format!("{}/Olsen/Door_Lund/",runtime_conf.dataset_path);
    let data_set_ahlströmer_path =  format!("{}/Olsen/Jonas_Ahlströmer/",runtime_conf.dataset_path);
    let data_set_fountain_path =  format!("{}/Olsen/fountain/",runtime_conf.dataset_path);
    let data_set_vasa_path =  format!("{}/Olsen/vasa_statue/",runtime_conf.dataset_path);
    let data_set_ninjo_path =  format!("{}/Olsen/nijo/",runtime_conf.dataset_path);
    let data_set_de_guerre_path =  format!("{}/Olsen/de_guerre/",runtime_conf.dataset_path);
    let data_set_fort_channing_path = format!("{}/Olsen/Fort_Channing_gate/",runtime_conf.dataset_path);
    
    let olsen_data_path = data_set_door_path;
    let depth_prior = -1.0;
    let epipolar_thresh = 0.01;
    //let epipolar_thresh = 0.01;

    
    let olsen_data = OlssenData::new(&olsen_data_path);
    let positive_principal_distance = false;
    let principal_distance_sign = match positive_principal_distance {
        true => 1.0,
        false => -1.0
    };
    let invert_intrinsics = false; // they are already negative from decomp
    let normalize_features = false;
    let feature_skip_count = 2;


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


    let matches_5_4 = olsen_data.get_matches_between_images(5, 4);
    let matches_5_6 = olsen_data.get_matches_between_images(5, 6);
    // let matches_5_7 = olsen_data.get_matches_between_images(5, 7);
    // let matches_5_8 = olsen_data.get_matches_between_images(5, 8);

    // let matches_6_0 = olsen_data.get_matches_between_images(6, 0);
    // let matches_6_1 = olsen_data.get_matches_between_images(6, 1);
    // let matches_6_2 = olsen_data.get_matches_between_images(6, 2);
    // let matches_6_3 = olsen_data.get_matches_between_images(6, 3);
    // let matches_6_4 = olsen_data.get_matches_between_images(6, 4);
    // let matches_6_5 = olsen_data.get_matches_between_images(6, 5);
    // let matches_6_7 = olsen_data.get_matches_between_images(6, 7);
    // let matches_6_8 = olsen_data.get_matches_between_images(6, 8);
    // let matches_6_9 = olsen_data.get_matches_between_images(6, 9);
    // let matches_6_10 = olsen_data.get_matches_between_images(6, 10);
    // let matches_6_11 = olsen_data.get_matches_between_images(6, 11); 
    //let matches_6_12 = olsen_data.get_matches_between_images(6, 12);
    //let matches_6_13 = olsen_data.get_matches_between_images(6, 13);
    //let matches_6_17 = olsen_data.get_matches_between_images(6, 17);
    //let matches_17_20 = olsen_data.get_matches_between_images(17, 20);


    let matches_5_4_subvec = matches_5_4.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    let matches_5_6_subvec = matches_5_6.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_5_7_subvec = matches_5_7.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_5_8_subvec = matches_5_8.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_6_5_subvec = matches_6_5.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_6_7_subvec = matches_6_7.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_6_8_subvec = matches_6_8.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_6_9_subvec = matches_6_9.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_6_10_subvec = matches_6_10.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    // let matches_6_11_subvec = matches_6_11.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    //let matches_6_12_subvec = matches_6_12.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    //let matches_6_13_subvec = matches_6_13.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    //let matches_6_17_subvec = matches_6_17.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();
    //let matches_17_20_subvec = matches_6_17.iter().enumerate().filter(|&(i,_)| i % feature_skip_count == 0).map(|(_,x)| x.clone()).collect::<Vec<Match<ImageFeature>>>();


    // let matches_0_5 = olsen_data.get_matches_between_images(0, 5);
    // println!("matches between 0 and 5 are: #{}", matches_0_5.len());
    
    // let matches_0_10 = olsen_data.get_matches_between_images(0, 10);camera_data
    // let matches_0_20 = olsen_data.get_matches_between_images(0, 20);
    // println!("matches between 0 and 20 are: #{}", matches_0_20.len());

    let pinhole_cam_0 = Perspective::from_matrix(&cam_intrinsics_0, invert_intrinsics);
    let pinhole_cam_1 = Perspective::from_matrix(&cam_intrinsics_1, invert_intrinsics);
    let pinhole_cam_2 = Perspective::from_matrix(&cam_intrinsics_2, invert_intrinsics);
    let pinhole_cam_3 = Perspective::from_matrix(&cam_intrinsics_3, invert_intrinsics);
    let pinhole_cam_4 = Perspective::from_matrix(&cam_intrinsics_4, invert_intrinsics);
    let pinhole_cam_5 = Perspective::from_matrix(&cam_intrinsics_5, invert_intrinsics);
    let pinhole_cam_6 = Perspective::from_matrix(&cam_intrinsics_6, invert_intrinsics);
    let pinhole_cam_7 = Perspective::from_matrix(&cam_intrinsics_7, invert_intrinsics);
    let pinhole_cam_8 = Perspective::from_matrix(&cam_intrinsics_8, invert_intrinsics);
    let pinhole_cam_9 = Perspective::from_matrix(&cam_intrinsics_9, invert_intrinsics);
    let pinhole_cam_10 = Perspective::from_matrix(&cam_intrinsics_10, invert_intrinsics);
    let pinhole_cam_11 = Perspective::from_matrix(&cam_intrinsics_11, invert_intrinsics);
    //let pinhole_cam_12 = Perspective::from_matrix(&cam_intrinsics_12, false);


    let sfm_all_matches = vec!(vec!(matches_5_4_subvec),vec!(matches_5_6_subvec)); //TODO: remove clone after refeactor!
    
    let mut camera_data = Vec::<((usize,Perspective),(usize,Perspective))>::with_capacity(10); 
    camera_data.push(((5,pinhole_cam_5),(4,pinhole_cam_4)));
    camera_data.push(((5,pinhole_cam_5),(6,pinhole_cam_6)));

    let camera_map = HashMap::from([(5, pinhole_cam_5), (4, pinhole_cam_4), (6, pinhole_cam_6)]);
    let paths = vec!(vec!(4),vec!(6));

    //camera_data.push(((6,pinhole_cam_6),(4,pinhole_cam_4)));
    //camera_data.push(((6,pinhole_cam_6),(5,pinhole_cam_5)));
    //camera_data.push(((6,pinhole_cam_6),(7,pinhole_cam_7)));
    //camera_data.push(((6,pinhole_cam_6),(8,pinhole_cam_8)));
    //camera_data.push(((6,pinhole_cam_6),(9,pinhole_cam_9)));
    //camera_data.push(((7,pinhole_cam_7),(7,pinhole_cam_8)));


    let sfm_config = SFMConfig::new(5, paths, camera_map, sfm_all_matches);
    let (initial_cam_motions_per_path,filtered_matches_per_path) = compute_pairwise_cam_motions_with_filtered_matches(
            &sfm_config,
            1.0,
            epipolar_thresh,
            positive_principal_distance,
            normalize_features,
            BifocalType::ESSENTIAL, 
            EssentialDecomposition::FÖRSNTER
    );

    //This is only to satisfy current interface in ba
    let initial_cam_motions = initial_cam_motions_per_path.into_iter().flatten().collect::<Vec<(usize,(Vector3<Float>,Matrix3<Float>))>>();
    let filtered_matches = filtered_matches_per_path.clone().into_iter().flatten().collect::<Vec<Vec<Match<ImageFeature>>>>();

    //TODO: might be unneccesary
    let mut motion_list =  Vec::<((usize,Matrix4<Float>),(usize,Matrix4<Float>))>::with_capacity(10); 
    motion_list.push(((5,cam_extrinsics_5),(4,cam_extrinsics_4)));
    //motion_list.push(((5,cam_extrinsics_5),(6,cam_extrinsics_6)));
    //motion_list.push(((6,cam_extrinsics_6),(1,cam_extrinsics_1)));
    // motion_list.push(((6,cam_extrinsics_6),(2,cam_extrinsics_2)));
    // motion_list.push(((6,cam_extrinsics_6),(3,cam_extrinsics_3)));
    // motion_list.push(((6,cam_extrinsics_6),(4,cam_extrinsics_4)));
    let relative_motions = OlssenData::get_relative_motions(&motion_list);


    //This is only to satisfy current interface
    for i in 0..filtered_matches.len() {
        let m = &filtered_matches[i];
        let m_orig = &sfm_config.matches()[i][0];
        println!("orig matches: {}, filtered matches: {}", m_orig.len(), m.len());
    }


    let initial_cam_poses = Some(initial_cam_motions);
    //let initial_cam_poses = Some(relative_motions);
    //let initial_cam_poses = None;

    if initial_cam_poses.is_some(){
        for (_,(t,r)) in initial_cam_poses.as_ref().unwrap() {
            println!("t : {}",t);
            println!("r : {}",r);
            println!("-------");
        }
    }

    let used_matches = &filtered_matches;

    if used_matches.len() > 0 {

        let runtime_parameters = RuntimeParameters {
            pyramid_scale: 1.0,
            max_iterations: vec![160; 1],
            eps: vec![1e-6],
            step_sizes: vec![1e0],
            max_norm_eps: 1e-30, 
            delta_eps: 1e-30,
            taus: vec![1e-3],
            lm: true,
            debug: true,
            show_octave_result: true,
            loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
            intensity_weighting_function:  Box::new(weighting::SquaredWeight {})
            //intensity_weighting_function:  Box::new(weighting::CauchyWeight {c: 0.01})
        };


        //TODO: Features are between adjacent cams, but transform is not. -> Mistake!
        let ((cam_positions,points),(s,debug_states_serialized)) = run_ba(&filtered_matches_per_path, &sfm_config, &camera_data,&initial_cam_poses, olsen_data.get_image_dim(), &runtime_parameters, 1.0,depth_prior);
        fs::write(format!("{}/olsen.txt",runtime_conf.local_data_path), s?).expect("Unable to write file");
        if runtime_parameters.debug {
            fs::write(format!("{}/olsen_debug.txt",runtime_conf.local_data_path), debug_states_serialized?).expect("Unable to write file");
        }
    }


    Ok(())
}