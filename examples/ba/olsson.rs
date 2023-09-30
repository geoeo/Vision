extern crate nalgebra as na;

use color_eyre::eyre::Result;
use std::fs;
use vision::io::olsson_loader::OlssenData;
use vision::sfm::{triangulation::Triangulation,SFMConfig, compute_path_pairs_as_vec, bundle_adjustment::run_ba, epipolar::tensor::BifocalType};
use vision::sfm::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::load_runtime_conf;
use vision::visualize;

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    println!("--------");

    let ceiling_barcelona = "Ceiling_Barcelona";
    let door = "Door_Lund";
    let ahlströmer = "Jonas_Ahlströmer";
    let fountain = "fountain";
    let vasa = "vasa_statue";
    let ninjo = "nijo";
    let de_guerre = "de_guerre";
    let fort_channing = "Fort_Channing_gate";
    let park_gate = "park_gate";
    let kronan = "kronan";
    let round_church = "round_church";

    let olsen_dataset_name = ninjo;
    let olsen_data_path = format!("{}/Olsson/{}/",runtime_conf.dataset_path,olsen_dataset_name);

    let feature_skip_count = 1;
    let olsen_data = OlssenData::new(&olsen_data_path);
    let positive_principal_distance = true;
    let invert_y = !positive_principal_distance;
    let invert_focal_length = false;
    let refince_rotation_via_rcd = true;


    // let paths = vec!(vec!(6));
    // let root_id = 5;

    // let paths = vec!(vec!(4),vec!(6));
    // let root_id = 5;

    // let paths = vec!(vec!(4,3));
    // let root_id = 5;
    
    let paths = vec!(vec!(6,7));
    let root_id = 5;

    // let paths = vec!(vec!(6,7,8));
    // let root_id = 5;

    // let paths = vec!(vec!(6,7,8,9,10));
    // let root_id = 5;

    // let paths = vec!(vec!(5),vec!(7));
    // let root_id = 6;

    // let paths = vec!(vec!(4),vec!(6,7));
    // let root_id = 5;

    // let paths = vec!(vec!(4,3),vec!(6));
    // let root_id = 5;

    // let paths = vec!(vec!(4,3),vec!(6));
    // let root_id = 5;

    // let paths = vec!(vec!(4),vec!(6,7));
    // let root_id = 5;

    // let paths = vec!(vec!(4,3,2));
    // let root_id = 5;

    // let paths = vec!(vec!(6,7));
    // let root_id = 5;

    // let paths = vec!(vec!(6,7,8,9,10,11));
    // let root_id = 5;

    // let paths = vec!(vec!(6,7,8,9,10,11,12,13,14));
    // let root_id = 5;

    // let paths = vec!(vec!(6,8,9,11,12,14));
    // let root_id = 5;

    // let paths = vec!(vec!(6,7,8,9,10,11));
    // let root_id = 5;

    // let paths = vec!(vec!(4,3,2,1),vec!(6,7,8,9));
    // let root_id = 5;

    // let paths = vec!(vec!(2,3));
    // let root_id = 1;

    // let paths = vec!(vec!(7,6),vec!(9,10,11));
    // let root_id = 8;

    // let paths = vec!(vec!(6,5),vec!(9,10));
    // let root_id = 8;

    // let paths = vec!(vec!(9,10,11,12,13));
    // let root_id = 8;


    //TODO: implement switch for loftr matches!
    let (match_map, camera_map) = olsen_data.get_data_for_sfm(root_id, &paths, positive_principal_distance, invert_focal_length, invert_y, feature_skip_count, olsen_dataset_name);
    let sfm_config_fundamental = SFMConfig::new(root_id, &paths, None, camera_map, &match_map, 
    BifocalType::FUNDAMENTAL, Triangulation::STEREO, 1.0, 2.0e0, 5e0, 5.0, refince_rotation_via_rcd, true);

    for (key, pose) in sfm_config_fundamental.pose_map().iter() {
        println!("Key: {:?}, Pose: {:?}", key, pose)
    }

    for (i,j) in compute_path_pairs_as_vec(sfm_config_fundamental.root(),sfm_config_fundamental.paths()).into_iter().flatten().collect::<Vec<_>>() {
        let im_1 = olsen_data.get_image(i);
        let im_2 = olsen_data.get_image(j);
        let matches = sfm_config_fundamental.match_map().get(&(i,j)).expect(format!("Match ({},{}) not present!",i,j).as_str());
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
        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {}),
        //intensity_weighting_function:  Box::new(weighting::HuberWeight {}),
        cg_threshold: 1e-6,
        cg_max_it: 2e3 as usize
    };

    let ((_,_),(s,debug_states_serialized)) = run_ba(&sfm_config_fundamental, &runtime_parameters);
    fs::write(format!("{}/ba.txt",runtime_conf.output_path), s?).expect("Unable to write file");
    if runtime_parameters.debug {
        fs::write(format!("{}/ba_debug.txt",runtime_conf.output_path), debug_states_serialized?).expect("Unable to write file");
    }

    Ok(())
}

