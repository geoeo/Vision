extern crate nalgebra as na;

use na::{DVector,Matrix3x4,Matrix4xX};
use crate::image::features::{matches::Match,Feature};
use crate::sensors::camera::Camera;
use crate::{float,Float};

pub mod dual;

pub fn calculate_reprojection_errors<Feat: Feature, C: Camera<Float>>(landmarks: &Matrix4xX<Float>, matches: &Vec<Match<Feat>>, transform_c1: &Matrix3x4<Float>, cam_1 :&C, transform_c2: &Matrix3x4<Float>, cam_2 :&C) -> DVector<Float> {
    let landmark_count = landmarks.ncols();
    let mut reprojection_errors = DVector::<Float>::zeros(landmark_count);

    for i in 0..landmarks.ncols() {
        let p = landmarks.fixed_columns::<1>(i).into_owned();
        let m = &matches[i];
        let feat_1 = &m.get_feature_one().get_as_2d_point();
        let feat_2 = &m.get_feature_two().get_as_2d_point();
        let p_cam_1 = transform_c1*p;
        let p_cam_2 = transform_c2*p;

        let projected_1 = cam_1.project(&p_cam_1);
        let projected_2 = cam_2.project(&p_cam_2);

        if projected_1.is_some() && projected_2.is_some() {
            let projected_1 = projected_1.unwrap().to_vector();
            let projected_2 = projected_2.unwrap().to_vector();
            reprojection_errors[i] = (feat_1-projected_1).norm() + (feat_2-projected_2).norm()
        } else {
            reprojection_errors[i] = float::INFINITY;
        }
    }
    reprojection_errors
}

pub fn calcualte_disparities<Feat: Feature>(matches: &Vec<Match<Feat>>) -> DVector<Float> {
    let mut disparities = DVector::<Float>::zeros(matches.len());
    for i in 0..matches.len() {
        let m = &matches[i];
        disparities[i] = (m.get_feature_one().get_as_2d_point() - m.get_feature_two().get_as_2d_point()).norm()
    }
    disparities
}