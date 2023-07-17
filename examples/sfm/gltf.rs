use color_eyre::eyre::Result;

use std::collections::{HashMap, HashSet};
use vision::{Float,float,load_runtime_conf};
use vision::sfm::{triangulation::Triangulation,SFMConfig, bundle_adjustment::run_ba, epipolar::tensor::BifocalType};
use vision::sensors::camera::perspective::Perspective;
use vision::image::features::{matches::Match,image_feature::ImageFeature};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();
    //let file_name = "camera_features_Suzanne_all.yaml";
    let file_name = "camera_features_Suzanne_x.yaml";
    //let file_name = "camera_features_Suzanne.yaml";
    //let file_name = "camera_features_Sphere.yaml";
    let path = format!("{}/{}",runtime_conf.local_data_path,file_name);
    let loaded_data = models_cv::io::deserialize_feature_matches(&path);


    let camera_map = loaded_data.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let intrinsic_matrix = cf.get_intrinsic_matrix().cast::<Float>();
        (cam_id,Perspective::<Float>::from_matrix(&intrinsic_matrix, true))
    }).collect::<HashMap<_,_>>();

    let feature_map = loaded_data.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let map = cf.get_feature_map();
        (cam_id,map)
    }).collect::<HashMap<_,_>>();

    //let camera_id_pairs = vec!((1,2));
    //let camera_id_pairs = vec!((0,1));
    let camera_id_pairs = vec!((0,1),(1,2));


    let match_map = camera_id_pairs.iter().map(|(id1,id2)| {
        let fm_1 = feature_map.get(id1).expect("Feature map for cam id not available!");
        let fm_2 = feature_map.get(id2).expect("Feature map for cam id not available!");

        let keys_1 = fm_1.keys().into_iter().map(|&id| id).collect::<HashSet<_>>();
        let keys_2 = fm_2.keys().into_iter().map(|&id| id).collect::<HashSet<_>>();

        let shared_keys = keys_1.intersection(&keys_2).collect::<HashSet<_>>();

        let matches = shared_keys.iter().map(|key| {
            let f_1 = fm_1.get(key).expect("Invalid key!");
            let f_2 = fm_2.get(key).expect("Invalid key!");
            let cam_1_height =  camera_map.get(id1).expect("Invalid key!").get_cy()*2.0;
            let cam_2_height =  camera_map.get(id2).expect("Invalid key!").get_cy()*2.0;

            assert!(cam_1_height.fract() == 0.0);
            assert!(cam_2_height.fract() == 0.0);

            //GLTF 2.0 is defined with a RHS along -Z. Point are exported as-is, so we flip them here
            let image_feature_1 = ImageFeature::new(f_1.x as Float, cam_1_height - 1.0 - (f_1.y as Float), None);
            let image_feature_2 = ImageFeature::new(f_2.x as Float, cam_2_height - 1.0 - (f_2.y as Float), None);

            Match::new(image_feature_1, image_feature_2)
        }).collect::<Vec<_>>();

        ((*id1,*id2),matches)

    }).collect::<HashMap<_,_>>();

    let paths = vec![camera_id_pairs.iter().map(|&(_,c)| c).collect::<Vec<_>>()];
    let root_id = camera_id_pairs[0].0;

    let sfm_config_fundamental = SFMConfig::new(root_id, &paths, camera_map, &match_map, 
        BifocalType::ESSENTIAL_RANSAC, Triangulation::LINEAR, 1.0, 1e2, 5e2, 1.0, true, true, true); // Investigate epipolar thresh -> more deterministic wither lower value?

    for (key, pose) in sfm_config_fundamental.pose_map().iter() {
        println!("Key: {:?}, Pose: {:?}", key, pose)
    }

    let runtime_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![5e4 as usize; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e0],
        max_norm_eps: 1e-30, 

        delta_eps: 1e-30,
        taus: vec![1.0e0],
        lm: true,
        debug: false,
        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss { eps: 1e-16, approximate_gauss_newton_matrices: false }), 
        intensity_weighting_function:  Box::new(weighting::SquaredWeight {}),
        cg_threshold: 1e-6,
        cg_max_it: 2e3 as usize
    };
    let (_,(s,_)) = run_ba(&sfm_config_fundamental, &runtime_parameters);
    std::fs::write(format!("{}/glft.txt",runtime_conf.output_path), s?).expect("Unable to write file");

    Ok(())
}