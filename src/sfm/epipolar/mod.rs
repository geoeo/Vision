extern crate nalgebra as na;
extern crate rand;

pub mod tensor;

use na::{Vector3, Matrix3};
use crate::Float;
use crate::image::features::{Feature,matches::Match};

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

/**
 * Computes the epipolar lines of a match.
 * Returns (line of first feature in second image, line of second feature in first image)
 */
pub fn epipolar_lines<T: Feature>(bifocal_tensor: &Matrix3<Float>, feature_match: &Match<T>, cam_one_intrinsics: &Matrix3<Float>, cam_two_intrinsics: &Matrix3<Float>) -> (Vector3<Float>, Vector3<Float>) {
    let f_from = feature_match.get_feature_one().get_camera_ray(cam_one_intrinsics);
    let f_to = feature_match.get_feature_two().get_camera_ray(cam_two_intrinsics);

    ((f_from.transpose()*bifocal_tensor).transpose(), bifocal_tensor*f_to)
}

