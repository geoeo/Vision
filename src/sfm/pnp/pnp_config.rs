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
    landmarks: Matrix4xX<Float>,
    features: Vec<Feat>,
    features_norm: Vec<Feat>
}
