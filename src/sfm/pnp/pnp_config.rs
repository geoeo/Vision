extern crate nalgebra as na;

use na::Isometry3;
use std::collections::HashMap;
use crate::image::features::{Feature, compute_linear_normalization};
use crate::sensors::camera::Camera;
use crate::Float;
use crate::sfm::landmark::euclidean_landmark::EuclideanLandmark;


pub struct PnPConfig<C, Feat: Feature> {
    camera: C,
    camera_norm: C,
    landmarks: Vec<EuclideanLandmark<Float>>,
    features: Vec<Feat>,
    features_norm: Vec<Feat>,
    camera_pose_option: Option<Isometry3<Float>>
}

impl<C: Camera<Float> + Clone, Feat: Feature> PnPConfig<C,Feat> {
    pub fn new(
        camera: &C,
        //Indexed by landmark id
        landmark_map: &HashMap<usize, EuclideanLandmark<Float>>,
        feature_map: &HashMap<usize, Feat>,
        camera_pose_option: &Option<Isometry3<Float>>
    ) -> PnPConfig<C, Feat> {
        let keys = feature_map.keys();
        let mut landmarks = Vec::<EuclideanLandmark<Float>>::with_capacity(keys.len());
        let mut features = Vec::<Feat>::with_capacity(keys.len());
        for key in keys {
            let l = landmark_map.get(key).unwrap();
            let f = feature_map.get(key).unwrap();
            features.push(f.clone());
            landmarks.push(l.clone());
        }

        let (norm, norm_inv) = compute_linear_normalization(&features);
        let camera_matrix_norm = norm * camera.get_projection();
        let inverse_camera_matrix_norm = camera.get_inverse_projection() * norm_inv;
        let c_norm: C = Camera::from_matrices(&camera_matrix_norm, &inverse_camera_matrix_norm);
        let features_norm = features.iter().map(|f| f.apply_normalisation(&norm, 1.0)).collect::<Vec<_>>();
        PnPConfig{
            camera: camera.clone(),
            camera_norm: c_norm,
            landmarks,
            features,
            features_norm: features_norm,
            camera_pose_option: camera_pose_option.clone()
        }
    }

    pub fn get_camera(&self) -> &C {
        &self.camera
    }

    pub fn get_camera_norm(&self) -> &C {
        &self.camera_norm
    }

    pub fn get_landmarks(&self) -> &Vec<EuclideanLandmark<Float>> {
        &self.landmarks
    }

    pub fn get_features(&self) -> &Vec<Feat> {
        &self.features
    }

    pub fn get_features_norm(&self) -> &Vec<Feat> {
        &self.features_norm
    }

    pub fn get_camera_pose_option(&self) ->  &Option<Isometry3<Float>> {
        &self.camera_pose_option
    }


}
