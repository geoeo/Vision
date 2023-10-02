extern crate nalgebra as na;
use color_eyre::eyre::Result;

use std::collections::HashMap;
use vision::{Float,load_runtime_conf};
use vision::sfm::{triangulation::Triangulation,pnp::pnp_config::PnPConfig, bundle_adjustment::run_ba, epipolar::tensor::BifocalType};
use vision::sensors::camera::perspective::Perspective;

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();

    //let file_name = "/trans_x/camera_features_Suzanne.yaml";
    //let file_name = "/trans_y/camera_features_Suzanne.yaml";
    //let file_name = "/trans_z/camera_features_Suzanne.yaml";
    let file_name = "/60_10/camera_features_Suzanne.yaml"; //Maybe too little translation

    //let file_name = "/trans_x/camera_features_sphere.yaml";
    //let file_name = "/trans_y/camera_features_sphere.yaml";
    //let file_name = "/trans_z/camera_features_sphere.yaml";
    //let file_name = "/60_10/camera_features_sphere.yaml"; //Maybe too little translation

    //let file_name = "/trans_x/camera_features_Cube.yaml";
    //let file_name = "/trans_y/camera_features_Cube.yaml";
    //let file_name = "/trans_z/camera_features_Cube.yaml";

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

    let camera_poses = loaded_data.iter().map(|cf|  {
        let cam_id = cf.get_cam_id();
        let map = cf.get_view_matrix().cast::<Float>();
        (cam_id,map)
    }).collect::<HashMap<_,_>>();


    let cam_id = 0;
    let camera = camera_map.get(&cam_id).unwrap();
    let feature_map = feature_map.get(&cam_id).unwrap();
    let camera_map = camera_poses.get(&cam_id).unwrap();
    //let landmark_map = 

    //let pnp_config = PnPConfig::new(camera, landmarks, features, &None);
    //TODO: Start Pnp


    Ok(())
}