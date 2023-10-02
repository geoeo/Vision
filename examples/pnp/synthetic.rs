extern crate nalgebra as na;
use color_eyre::eyre::Result;

use std::collections::HashMap;
use vision::{Float,load_runtime_conf};
use vision::sfm::{triangulation::Triangulation,pnp::pnp_config::PnPConfig, bundle_adjustment::run_ba, epipolar::tensor::BifocalType};
use vision::sensors::camera::perspective::Perspective;

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    let scenario = "60_10"; //Maybe too little translation
    //let scenario = "trans_x";
    //let scenario = "trans_y";
    //let scenario = "trans_z";
    let dataset = "Suzanne";
    //let dataset = "sphere";
    //let dataset = "Cube";

    let cam_features_path = format!("{}/{}/camera_features_{}",runtime_conf.local_data_path,scenario,dataset);
    let landmark_path = format!("{}/{}/landmarks_{}",runtime_conf.local_data_path,scenario,dataset);

    let loaded_data_cam = models_cv::io::deserialize_feature_matches(&cam_features_path);
    let loaded_data_landmarks = models_cv::io::deserialize_landmarks(&landmark_path);


    let camera_map = loaded_data_cam.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let intrinsic_matrix = cf.get_intrinsic_matrix().cast::<Float>();
        (cam_id,Perspective::<Float>::from_matrix(&intrinsic_matrix, true))
    }).collect::<HashMap<_,_>>();

    let feature_map = loaded_data_cam.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let map = cf.get_feature_map();
        (cam_id,map)
    }).collect::<HashMap<_,_>>();

    let camera_poses = loaded_data_cam.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let map = cf.get_view_matrix().cast::<Float>();
        (cam_id,map)
    }).collect::<HashMap<_,_>>();


    let cam_id = 0;
    let camera = camera_map.get(&cam_id).unwrap();
    let feature_map = feature_map.get(&cam_id).unwrap();
    let camera_map = camera_poses.get(&cam_id).unwrap();
    let landmark_map = loaded_data_landmarks.iter().map(|l| (*l.get_id(), *l.get_position())).collect::<HashMap<_,_>>();

    //let pnp_config = PnPConfig::new(camera, landmarks, features, &None);
    //TODO: Start Pnp


    Ok(())
}