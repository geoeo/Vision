
extern crate nalgebra as na;

use na::{Matrix3,Vector2,DMatrix,DVector,Matrix4xX,Isometry3};
use std::collections::{HashMap,HashSet};
use crate::image::features::Feature;
use crate::Float;


/**
 * Outlier Rejection Using Duality Olsen et al.
 */
pub fn outlier_rejection_dual<Feat: Feature + Clone>(unique_landmark_ids: &HashSet<usize>, camera_ids_root_first: &Vec<usize>, abs_pose_map: &HashMap<usize,Isometry3<Float>>, feature_map: &HashMap<usize, HashSet<Feat>>) {
    assert_eq!(camera_ids_root_first.len(),abs_pose_map.keys().len());
    assert_eq!(camera_ids_root_first.len(),feature_map.keys().len());
    assert!(unique_landmark_ids.contains(&0)); // ids have to represent matrix indices

    let (a,b,c,a0,b0,c0) = generate_known_rotation_problem(unique_landmark_ids, camera_ids_root_first, abs_pose_map, feature_map);
    panic!("TODO");
}

fn generate_known_rotation_problem<Feat: Feature + Clone>(unique_landmark_ids: &HashSet<usize>, camera_ids_root_first: &Vec<usize>, abs_pose_map: &HashMap<usize,Isometry3<Float>>, feature_map: &HashMap<usize, HashSet<Feat>>) -> (DMatrix<Float>,DMatrix<Float>,DMatrix<Float>,DVector<Float>,DVector<Float>,DVector<Float>) {
    let number_of_unique_points = unique_landmark_ids.len();
    let number_of_poses = abs_pose_map.len();
    let number_of_target_parameters = 3*number_of_unique_points + 3*(number_of_poses-1); // The first pose is taken as identity (origin) hence we dont optimize it
    let number_of_residuals = feature_map.values().fold(0, |acc, x| acc + x.len());

    let mut a = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut b = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut c = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);

    let mut a0 = DVector::<Float>::zeros(number_of_residuals);
    let mut b0 = DVector::<Float>::zeros(number_of_residuals);
    let mut c0 = DVector::<Float>::zeros(number_of_residuals);

    for i in 0..number_of_poses {
        let cam_id = camera_ids_root_first[i];
        let rotation = abs_pose_map.get(&cam_id).expect("generate_known_rotation_problem: No rotation found").rotation.to_rotation_matrix();
        let rotation_matrix = rotation.matrix();
        let features = feature_map.get(&cam_id).expect("generate_known_rotation_problem: No features found");
        let number_of_points = features.len(); // asuming every feature's landmark id is distinct -> maybe make an explicit check?
        let rows = 3*number_of_points*2;
        let ones = DVector::<Float>::repeat(number_of_points, 1.0);

        let mut p = DVector::<Float>::zeros(number_of_points);
        let mut p_idx = 0;
        for feature in features {
            p[p_idx] = feature.get_x_image_float();
            p_idx += 1;
        }
        let point_coeff = (&p)*rotation_matrix.row(2) - (&ones)*rotation_matrix.row(0);
        let mut t_coeff = DVector::<Float>::zeros(number_of_points);
        t_coeff.column_mut(0).copy_from(&(-(&ones)));
        t_coeff.column_mut(2).copy_from(&p);






        //TODO feature point index matters as the first N columns are landmarks

        
    }


    //TODO
    (a,b,c,a0,b0,c0)
}