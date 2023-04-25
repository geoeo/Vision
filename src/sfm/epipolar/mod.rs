extern crate nalgebra as na;
extern crate rand;

pub mod tensor;

use na::{Vector2, Vector3, Matrix3};
use crate::Float;
use crate::image::features::{Feature,matches::Match};

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

pub fn compute_linear_normalization<Feat: Feature>(features: &Vec<Feat>) -> (Matrix3<Float>, Matrix3<Float>) {
    let l = features.len();
    let l_as_float = l as Float;

    let mut normalization_matrix = Matrix3::<Float>::identity();
    let mut normalization_matrix_inv= Matrix3::<Float>::identity();

    let mut avg = Vector2::<Float>::zeros();

    for feat in features{
        let f = feat.get_as_2d_point();
        avg[0] += f[0];
        avg[1] += f[1];
    }

    avg /= l_as_float;
    avg /= l_as_float;

    let dist_mean_norm =  features.iter().fold(0.0, |acc, f| (acc + (f.get_as_2d_point()-avg).norm_squared()));

    let sqrt_2 = (2.0 as Float).sqrt();
    let s = (sqrt_2*l_as_float)/dist_mean_norm.sqrt();

    normalization_matrix[(0,0)] = s;
    normalization_matrix[(1,1)] = s;
    normalization_matrix[(0,2)] = -s*avg[0];
    normalization_matrix[(1,2)] = -s*avg[1];

    normalization_matrix_inv[(0,0)] = 1.0/s;
    normalization_matrix_inv[(1,1)] = 1.0/s;
    normalization_matrix_inv[(0,2)] = avg[0];
    normalization_matrix_inv[(1,2)] = avg[1];


    (normalization_matrix, normalization_matrix_inv)

}

/**
 * Computes the epipolar lines of a match.
 * Returns (line of first feature in second image, line of second feature in first image)
 */
pub fn epipolar_lines<T: Feature>(bifocal_tensor: &Matrix3<Float>, feature_match: &Match<T>, cam_one_intrinsics: &Matrix3<Float>, cam_two_intrinsics: &Matrix3<Float>, positive_principal_distance: bool) -> (Vector3<Float>, Vector3<Float>) {
    let f_from = feature_match.get_feature_one().get_camera_ray(cam_one_intrinsics, positive_principal_distance);
    let f_to = feature_match.get_feature_two().get_camera_ray(cam_two_intrinsics, positive_principal_distance);

    ((f_from.transpose()*bifocal_tensor).transpose(), bifocal_tensor*f_to)
}

