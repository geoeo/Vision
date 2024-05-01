extern crate nalgebra as na;

use color_eyre::eyre::Result;
use std::fs;
use vision::io::olsson_loader::OlssenData;
use vision::sfm::{
    triangulation::Triangulation,
    bundle_adjustment::ba_config::{BAConfig, conversions::compute_path_id_pairs},
    bundle_adjustment::run_ba, 
    epipolar::tensor::BifocalType, 
    pnp::run_pnp,runtime_parameters::RuntimeParameters,
    bundle_adjustment::ba_config::filtering::filter_config,
    state::landmark::inverse_depth_landmark::InverseLandmark
};
use vision::numerics::weighting;
use vision::load_runtime_conf;
use vision::visualize;
use vision::Float;

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    println!("--------");

    let ceiling_barcelona = "Ceiling_Barcelona";
    let door = "Door_Lund";
    let ahlströmer = "Jonas_Ahlströmer";
    let fountain = "fountain";
    let vasa = "vasa_statue"; // transposed features
    let nijo = "nijo";
    let de_guerre = "de_guerre";
    let fort_channing = "Fort_Channing_gate";
    let park_gate = "park_gate";
    let kronan = "kronan";
    let round_church = "round_church";

    let olsen_dataset_name = park_gate;
    let olsen_data_path = format!("{}/Olsson/{}/",runtime_conf.dataset_path,olsen_dataset_name);

    let feature_skip_count = 1;
    let olsen_data = OlssenData::new(&olsen_data_path);
    let positive_principal_distance = true;
    let invert_y = !positive_principal_distance;
    let invert_focal_length = false;
    let refince_rotation_via_rcd = true;
    let mut transpose_features = false;


    // let paths = vec!(vec!(6));
    // let root_id = 5;

    // let paths = vec!(vec!(4),vec!(6));
    // let root_id = 5;

    // let paths = vec!(vec!(4,3));
    // let root_id = 5;
    
    // let paths = vec!(vec!(6,7));
    // let root_id = 5;

    // door 
    // let paths = vec!(vec!(5,6));
    // let root_id = 4;

    // de_guerre
    // let paths = vec!(vec!(5,6));
    // let root_id = 4;

    // nijo
    // let paths = vec!(vec!(5,6,7,8));
    // let root_id = 4;

    // fountain
    // let paths = vec!(vec!(5,6,7,8));
    // let root_id = 4;

    // park gate 
    let paths = vec!(vec!(5,6,7,8));
    let root_id = 4;

    // kronan
    // let paths = vec!(vec!(5,6,7,8));
    // let root_id = 4;
    
    // vasa
    // transpose_features = true;
    // let paths = vec!(vec!(5,6,7,8,9,10));
    // let root_id = 4;

    // Fort Channing
    // let paths = vec!(vec!(8,10));
    // let root_id = 7;

    // // Round Church
    // let paths = vec!(vec!(7,8,10,11,12,13));
    // let root_id = 5;


    //TODO: implement switch for loftr matches!
    let (match_map, camera_map, image_width, image_height) = olsen_data.get_data_for_sfm(root_id, &paths, positive_principal_distance, invert_focal_length, invert_y, transpose_features, feature_skip_count, olsen_dataset_name);
    let mut ba_config_fundamental = BAConfig::new(root_id, &paths, None, camera_map, &match_map, 
    BifocalType::FUNDAMENTAL, Triangulation::LOST, 1.0, 1.0e0, image_width, image_height);
    let mut ba_config_fundamental = filter_config(&mut ba_config_fundamental,1e4, false, refince_rotation_via_rcd, Triangulation::LOST);

    for (key, pose) in ba_config_fundamental.pose_map().iter() {
        println!("Key: {:?}, Pose: {:?}", key, pose)
    }

    for (i,j) in compute_path_id_pairs(ba_config_fundamental.root(),ba_config_fundamental.paths()).into_iter().flatten().collect::<Vec<_>>() {
        let im_1 = olsen_data.get_image(i);
        let im_2 = olsen_data.get_image(j);
        let matches = ba_config_fundamental.match_map().get(&(i,j)).expect(format!("Match ({},{}) not present!",i,j).as_str());
        let vis_matches = visualize::display_matches_for_pyramid(im_1,im_2,&matches,true,125.0,1.0, invert_y);
        vis_matches.to_image().save(format!("{}/olsen_matches_{}_{}_{}.jpg",runtime_conf.output_path,olsen_dataset_name,i,j)).unwrap();
    }

    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![5e4 as usize; 1],
        eps: vec![1e-2],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1.0e0],
        lm: true,
        debug: false,
        print: true,
        show_octave_result: true,
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {}),
        //intensity_weighting_function:  Box::new(weighting::HuberWeight {}), 
        cg_threshold: 1e-6,
        cg_max_it: 2e3 as usize
    };

    let runtime_parameters_pnp = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![1e3 as usize; 1],
        eps: vec![1e-8],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 
        delta_eps: 1e-30,
        taus: vec![1.0e0],
        lm: true,
        debug: false,
        print: true,
        show_octave_result: true,
        intensity_weighting_function:  Box::new(weighting::HuberWeight {}),
        cg_threshold: 1e-6,
        cg_max_it: 2e3 as usize
    };


    let trajectories = compute_path_id_pairs(ba_config_fundamental.root(), ba_config_fundamental.paths());
    let camera_ids = ba_config_fundamental.camera_map().clone();

    for &cam_id in camera_ids.keys() {
        let pnp_config_cam = ba_config_fundamental.generate_pnp_config_from_cam_id(cam_id);
        let (optimized_state_pnp, _) = run_pnp(&pnp_config_cam,&runtime_parameters_pnp);
        ba_config_fundamental.update_state(&optimized_state_pnp);
        let cam_pos_pnp = optimized_state_pnp.get_camera_positions().first().unwrap().clone();
        println!("Cam state pnp: {}", cam_pos_pnp);
    }

    let (optimized_state, state_debug_list) = run_ba::<_,6,InverseLandmark<Float>,_,_>(&ba_config_fundamental, &runtime_parameters,&trajectories);
    let state_serialized = serde_yaml::to_string(&optimized_state.to_serial());
    let debug_states_serialized = serde_yaml::to_string(&state_debug_list);

    fs::write(format!("{}/sfm.txt",runtime_conf.output_path), state_serialized?).expect("Unable to write file");
    if runtime_parameters.debug {
        fs::write(format!("{}/sfm_debug.txt",runtime_conf.output_path), debug_states_serialized?).expect("Unable to write file");
    }

    Ok(())
}

