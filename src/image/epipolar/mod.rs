extern crate nalgebra as na;
extern crate rand;

pub mod tensor;

use na::{Vector3, Matrix3};
use crate::sensors::camera::Camera;
use crate::Float;
use crate::image::features::{Feature,Match, ImageFeature};

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

//TODO: Doenst work!
pub fn compute_linear_normalization<T: Feature, C: Camera<Float>>(matches: &Vec<Match<T>>, camera_one: &C, camera_two: &C) -> (Matrix3<Float>, Matrix3<Float>) {
    let l = matches.len();
    let l_as_float = l as Float;

    let mut normalization_matrix_one = Matrix3::<Float>::identity();
    let mut normalization_matrix_two = Matrix3::<Float>::identity();

    let mut avg_x_one = 0.0;
    let mut avg_y_one = 0.0;
    let mut avg_x_two = 0.0;
    let mut avg_y_two = 0.0;

    let projection_one =camera_one.get_projection();
    let projection_two =camera_two.get_projection();

    let cx_one = projection_one[(0,2)];
    let cy_one = projection_one[(1,2)];
    let cx_two = projection_two[(0,2)];
    let cy_two = projection_two[(1,2)];


    for i in 0..l {
        let m = &matches[i];
        let f_1 = m.feature_one.get_as_3d_point(-1.0);
        let f_2 = m.feature_two.get_as_3d_point(-1.0);
        avg_x_one += f_1[0];
        avg_y_one += f_1[1];
        avg_x_two += f_2[0];
        avg_y_two += f_2[1];
    }

    let max_dist_one = cx_one*cy_one;
    let max_dist_two = cx_two*cy_two;

    // let max_dist_one = cx_one.powi(2)+cy_one.powi(2);
    // let max_dist_two = cx_two.powi(2)+cy_two.powi(2);

    
    normalization_matrix_one[(0,2)] = -avg_x_one/l_as_float;
    normalization_matrix_one[(1,2)] = -avg_y_one/l_as_float;
    normalization_matrix_one[(2,2)] = max_dist_one;

    normalization_matrix_two[(0,2)] = -avg_x_two/l_as_float;
    normalization_matrix_two[(1,2)] = -avg_y_two/l_as_float;
    normalization_matrix_two[(2,2)] = max_dist_two;

    (normalization_matrix_one,normalization_matrix_two)

}

pub fn extract_matches<T: Feature>(matches: &Vec<Match<T>>, pyramid_scale: Float) -> Vec<Match<ImageFeature>> {
    matches.iter().map(|feature| {
        let (r_x, r_y) = feature.feature_two.reconstruct_original_coordiantes_for_float(pyramid_scale);
        let (l_x, l_y) = feature.feature_one.reconstruct_original_coordiantes_for_float(pyramid_scale);
        Match { feature_one: ImageFeature::new(l_x,l_y), feature_two: ImageFeature::new(r_x,r_y)}
    }).collect()
}

#[allow(non_snake_case)]
pub fn filter_matches_from_motion<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, relative_motion: &(Vector3<Float>,Matrix3<Float>),camera_pair: &(C,C), epipiolar_thresh: Float) -> Vec<Match<T>> {
    let (cam_s,cam_f) = &camera_pair;
    let (t,R) = &relative_motion;
    let essential = tensor::essential_matrix_from_motion(t, R);
    let cam_s_inv = cam_s.get_inverse_projection();
    let cam_f_inv = cam_f.get_inverse_projection();
    let fundamental = tensor::compute_fundamental(&essential, &cam_s_inv, &cam_f_inv);

    tensor::filter_matches_from_fundamental(&fundamental,matches, epipiolar_thresh, cam_s,cam_f)
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

