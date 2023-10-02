extern crate nalgebra as na;
extern crate num_traits;


use na::{DVector, Matrix4xX, Matrix3, Matrix4, Isometry3};
use std::{collections::{HashMap,HashSet}, hash::Hash};
use crate::image::features::{Feature, compute_linear_normalization, matches::Match, feature_track::FeatureTrack, solver_feature::SolverFeature};
use crate::sfm::{epipolar::tensor, 
    triangulation::{Triangulation, triangulate_matches}, 
    rotation_avg::optimize_rotations_with_rcd,
    outlier_rejection::dual::outlier_rejection_dual};
use crate::sfm::outlier_rejection::{calculate_reprojection_errors,calcualte_disparities, reject_landmark_outliers, filter_by_rejected_landmark_ids, reject_matches_via_disparity, compute_continuous_landmark_ids_from_matches,compute_continuous_landmark_ids_from_unique_landmarks};
use crate::sensors::camera::Camera;
use crate::numerics::pose::{from_matrix,se3};
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
        landmarks: Matrix4xX<Float>, //TODO: Maybe rework to also have an id. Check feature to landmark id
        features: Vec<Feat>,
        camera_pose_option: &Option<Isometry3<Float>>
    ) -> PnPConfig<C, Feat> {

        let (norm, norm_inv) = compute_linear_normalization(&features);
        let camera_matrix_norm = norm * camera.get_projection();
        let inverse_camera_matrix_norm = camera.get_inverse_projection() * norm_inv;
        let c_norm: C = Camera::from_matrices(&camera_matrix_norm, &inverse_camera_matrix_norm);
        let features_norm = features.iter().map(|f| f.apply_normalisation(&norm, 1.0)).collect::<Vec<_>>();
        PnPConfig{
            camera: camera.clone(),
            camera_norm: c_norm,
            landmarks: landmarks.clone(),
            features: features.clone(),
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
