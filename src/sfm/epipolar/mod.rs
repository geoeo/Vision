extern crate nalgebra as na;
extern crate rand;

pub mod tensor;

use na::{Vector2, Vector3, Matrix3};
use crate::sensors::camera::Camera;
use crate::Float;
use crate::image::features::{Feature,Match, ImageFeature};

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

pub fn compute_linear_normalization<T: Feature>(matches: &Vec<Match<T>>) -> (Matrix3<Float>, Matrix3<Float>, Matrix3<Float>, Matrix3<Float>) {
    let l = matches.len();
    let l_as_float = l as Float;

    let mut normalization_matrix_one = Matrix3::<Float>::identity();
    let mut normalization_matrix_two = Matrix3::<Float>::identity();

    let mut normalization_matrix_one_inv = Matrix3::<Float>::identity();
    let mut normalization_matrix_two_inv = Matrix3::<Float>::identity();

    let mut avg_one = Vector2::<Float>::zeros();
    let mut avg_two = Vector2::<Float>::zeros();

    for i in 0..l {
        let m = &matches[i];
        let f_1 = m.feature_one.get_as_2d_point();
        let f_2 = m.feature_two.get_as_2d_point();
        avg_one[0] += f_1[0];
        avg_one[1] += f_1[1];
        avg_two[0] += f_2[0];
        avg_two[1] += f_2[1];
    }

    avg_one /= l_as_float;
    avg_two /= l_as_float;

    let (avg_dist_one, avg_dist_two) =  matches.iter().fold((0.0, 0.0), |acc, m| (acc.0 + (m.feature_one.get_as_2d_point()-avg_one).norm_squared(), acc.1 + (m.feature_two.get_as_2d_point()-avg_two).norm_squared()));
    let sqrt_factor = (2.0 as Float).sqrt();

    //TODO: Move landmark id to track generation. Currently scaled pixels map to very small region e.g 5x5 region
    let s_one = 1.0/(avg_dist_one/(l_as_float-1.0)).sqrt();
    let s_two = 1.0/(avg_dist_two/(l_as_float-1.0)).sqrt();

    let s_one = 1.0;
    let s_two = 1.0;

    normalization_matrix_one[(0,0)] = s_one;
    normalization_matrix_one[(1,1)] = s_one;
    normalization_matrix_one[(0,2)] = -s_one*avg_one[0];
    normalization_matrix_one[(1,2)] = -s_one*avg_one[1];

    normalization_matrix_one_inv[(0,0)] = 1.0/s_one;
    normalization_matrix_one_inv[(1,1)] = 1.0/s_one;
    normalization_matrix_one_inv[(0,2)] = avg_one[0];
    normalization_matrix_one_inv[(1,2)] = avg_one[1];

    normalization_matrix_two[(0,0)] = s_two;
    normalization_matrix_two[(1,1)] = s_two;
    normalization_matrix_two[(0,2)] = -s_two*avg_two[0];
    normalization_matrix_two[(1,2)] = -s_two*avg_two[1];

    normalization_matrix_two_inv[(0,0)] = 1.0/s_two;
    normalization_matrix_two_inv[(1,1)] = 1.0/s_two;
    normalization_matrix_two_inv[(0,2)] = avg_two[0];
    normalization_matrix_two_inv[(1,2)] = avg_two[1];

    (normalization_matrix_one, normalization_matrix_one_inv, normalization_matrix_two, normalization_matrix_two_inv)

}

#[allow(non_snake_case)]
pub fn filter_matches_from_motion<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, relative_motion: &(Vector3<Float>,Matrix3<Float>),camera_pair: &(C,C), epipiolar_thresh: Float) -> Vec<Match<T>> {
    let (cam_s,cam_f) = &camera_pair;
    let (t,R) = &relative_motion;
    let essential = tensor::essential_matrix_from_motion(t, R);
    let cam_s_inv = cam_s.get_inverse_projection();
    let cam_f_inv = cam_f.get_inverse_projection();
    let fundamental = tensor::compute_fundamental(&essential, &cam_s_inv, &cam_f_inv);

    tensor::filter_matches_from_fundamental(&fundamental,matches, epipiolar_thresh)
}

/**
 * Computes the epipolar lines of a match.
 * Returns (line of first feature in second image, line of second feature in first image)
 */
pub fn epipolar_lines<T: Feature>(bifocal_tensor: &Matrix3<Float>, feature_match: &Match<T>, cam_one_intrinsics: &Matrix3<Float>, cam_two_intrinsics: &Matrix3<Float>) -> (Vector3<Float>, Vector3<Float>) {
    let f_from = feature_match.feature_one.get_camera_ray(cam_one_intrinsics);
    let f_to = feature_match.feature_two.get_camera_ray(cam_two_intrinsics);

    ((f_from.transpose()*bifocal_tensor).transpose(), bifocal_tensor*f_to)
}

