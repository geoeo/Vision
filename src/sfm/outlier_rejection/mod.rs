
extern crate nalgebra as na;

use na::{Matrix3,Vector2,DMatrix,DVector,Matrix4xX,Isometry3};
use std::collections::{HashMap,HashSet};
use crate::image::features::Feature;
use crate::Float;


/**
 * Outlier Rejection Using Duality Olsen et al.
 */
pub fn outlier_rejection_dual<Feat: Feature + Clone>(unique_landmark_ids: &HashSet<usize>, abs_pose_map: &HashMap<usize,Isometry3<Float>>, feature_map: &HashMap<usize, HashSet<Feat>>) {
    let (a,b,c,a0,b0,c0) = generate_known_rotation_problem(unique_landmark_ids,abs_pose_map,feature_map);
    panic!("TODO");
}

fn generate_known_rotation_problem<Feat: Feature + Clone>(unique_landmark_ids: &HashSet<usize>, abs_pose_map: &HashMap<usize,Isometry3<Float>>, feature_map: &HashMap<usize, HashSet<Feat>>) -> (DMatrix<Float>,DMatrix<Float>,DMatrix<Float>,DVector<Float>,DVector<Float>,DVector<Float>) {
    let number_of_unique_ids = unique_landmark_ids.len();
    let number_of_poses = abs_pose_map.len();
    let number_of_target_parameters = 3*number_of_unique_ids + 3*(number_of_poses-1); // The first pose is taken as identity (origin) hence we dont optimize it
    let number_of_residuals = feature_map.values().fold(0, |acc, x| acc + x.len());

    let mut a = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut b = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut c = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);

    let mut a0 = DVector::<Float>::zeros(number_of_residuals);
    let mut b0 = DVector::<Float>::zeros(number_of_residuals);
    let mut c0 = DVector::<Float>::zeros(number_of_residuals);



    //TODO
    (a,b,c,a0,b0,c0)
}