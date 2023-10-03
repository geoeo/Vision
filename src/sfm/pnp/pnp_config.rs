extern crate nalgebra as na;
extern crate num_traits;


use na::{Vector3, Matrix4xX, Matrix3, Matrix4, Isometry3};
use std::{collections::{HashMap,HashSet}, hash::Hash};
use crate::image::features::{Feature, compute_linear_normalization, solver_feature::SolverFeature};
use crate::sensors::camera::Camera;
use crate::{float,Float};


pub struct PnPConfig<C, Feat: Feature> {
    camera: C,
    camera_norm: C,
    landmarks: Matrix4xX<Float>,
    features: Vec<Feat>,
    features_norm: Vec<Feat>,
    camera_pose_option: Option<Isometry3<Float>>
}

impl<C: Camera<Float> + Clone, Feat: Feature + Clone + PartialEq + Eq + Hash + SolverFeature> PnPConfig<C,Feat> {
    pub fn new(
        camera: &C,
        landmark_map: &HashMap<usize, Vector3<Float>>, //TODO: Maybe rework to also have an id. Check feature to landmark id
        feature_map: &HashMap<usize, Feat>,
        camera_pose_option: &Option<Isometry3<Float>>
    ) -> PnPConfig<C, Feat> {
        let keys = feature_map.keys();
        let mut landmarks = Matrix4xX::<Float>::from_element(keys.len(),1.0);
        let mut features = Vec::<Feat>::with_capacity(keys.len());
        for (i,key) in keys.enumerate() {
            let l = landmark_map.get(key).unwrap();
            let f = feature_map.get(key).unwrap();
            features.push(f.clone());
            landmarks.fixed_view_mut::<3,1>(0, i).copy_from(l);
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

    pub fn get_landmarks(&self) -> &Matrix4xX<Float> {
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
