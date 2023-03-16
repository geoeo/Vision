
extern crate nalgebra as na;

use na::{MatrixXx3,Vector2,DMatrix,DVector,Isometry3, RowVector3};
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

fn sovlve_feasability_problem(a: &DMatrix<Float>, b: &DMatrix<Float>, c: &DMatrix<Float>, a0: &DVector<Float>, b0: &DVector<Float>, c0: &DVector<Float>, tol: Float, min_depth: Float, max_depth: Float) -> (DVector<Float>, DVector<Float>) {
    let mut a1 = DMatrix::<Float>::zeros(2*a.nrows() + 2*b.nrows(),a.ncols());
    a1.slice_mut((0, 0),a.shape()).copy_from(&(-a-tol*c));
    a1.slice_mut((a.nrows(), 0),a.shape()).copy_from(&(a-tol*c));
    a1.slice_mut((2*a.nrows(), 0),b.shape()).copy_from(&(-b-tol*c));
    a1.slice_mut((2*a.nrows() + b.nrows(), 0),b.shape()).copy_from(&(b-tol*c));

    let mut b1 = DVector::<Float>::zeros(2*a0.nrows() + 2*b0.nrows());
    b1.rows_mut(0, a0.nrows()).copy_from(&(a0+tol*c0));
    b1.rows_mut(a0.nrows(), a0.nrows()).copy_from(&(-a0+tol*c0));
    b1.rows_mut(2*a0.nrows(), b0.nrows()).copy_from(&(b0+tol*c0));
    b1.rows_mut(2*a0.nrows() + b0.nrows(), b0.nrows()).copy_from(&(-b0+tol*c0));


    panic!("TODO")
}

fn generate_known_rotation_problem<Feat: Feature + Clone>(unique_landmark_ids: &HashSet<usize>, camera_ids_root_first: &Vec<usize>, abs_pose_map: &HashMap<usize,Isometry3<Float>>, feature_map: &HashMap<usize, HashSet<Feat>>) -> (DMatrix<Float>,DMatrix<Float>,DMatrix<Float>,DVector<Float>,DVector<Float>,DVector<Float>) {
    let number_of_unique_points = unique_landmark_ids.len();
    let number_of_poses = abs_pose_map.len();
    let number_of_target_parameters = 3*number_of_unique_points + 3*(number_of_poses-1); // The first translation is taken as identity (origin) hence we dont optimize it
    let number_of_residuals = feature_map.values().fold(0, |acc, x| acc + x.len());

    let mut a = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut b = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut c = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);

    let a0 = DVector::<Float>::zeros(number_of_residuals);
    let b0 = DVector::<Float>::zeros(number_of_residuals);
    let c0 = DVector::<Float>::zeros(number_of_residuals);

    let mut row_acc = 0;
    // Skip root cam since we assume origin (TOOD: Check this)
    for cam_idx in 1..number_of_poses {
        let cam_id = camera_ids_root_first[cam_idx];
        let rotation = abs_pose_map.get(&cam_id).expect("generate_known_rotation_problem: No rotation found").rotation.to_rotation_matrix();
        let rotation_matrix = rotation.matrix();
        let features = feature_map.get(&cam_id).expect("generate_known_rotation_problem: No features found");
        let number_of_points = features.len(); // asuming every feature's landmark id is distinct -> maybe make an explicit check?
        let ones = DVector::<Float>::repeat(number_of_points, 1.0);

        let mut p_data_x = DVector::<Float>::zeros(number_of_points);
        let mut p_data_y = DVector::<Float>::zeros(number_of_points);
        let mut p_col_ids = DVector::<usize>::zeros(number_of_points);
        for (idx, feature) in features.iter().enumerate() {
            p_data_x[idx] = feature.get_x_image_float();
            p_data_y[idx] = feature.get_y_image_float();
            p_col_ids[idx] = feature.get_landmark_id().expect("generate_known_rotation_problem: no landmark id");
        }
        let point_coeff_x = (&p_data_x)*rotation_matrix.row(2) - (&ones)*rotation_matrix.row(0);
        let point_coeff_y = (&p_data_y)*rotation_matrix.row(2) - (&ones)*rotation_matrix.row(1); 
        let point_coeff_z = (&ones)*rotation_matrix.row(2); 

        let mut t_coeff_x = MatrixXx3::<Float>::zeros(number_of_points);
        let mut t_coeff_y = MatrixXx3::<Float>::zeros(number_of_points);
        let mut t_coeff_z = MatrixXx3::<Float>::zeros(number_of_points);
        t_coeff_x.column_mut(0).copy_from(&(-(&ones)));
        t_coeff_y.column_mut(0).copy_from(&(-(&ones)));
        t_coeff_x.column_mut(2).copy_from(&p_data_x);
        t_coeff_y.column_mut(2).copy_from(&p_data_y);
        t_coeff_z.column_mut(2).copy_from(&ones);

        for p_idx in 0..number_of_points{
            let v_p_x = point_coeff_x.row(p_idx);
            let v_p_y = point_coeff_y.row(p_idx);
            let v_p_z = point_coeff_z.row(p_idx);
            let v_t_x = t_coeff_x.row(p_idx);
            let v_t_y = t_coeff_y.row(p_idx);
            let v_t_z = t_coeff_z.row(p_idx);
            let col_id = p_col_ids[p_idx];
            let r_idx = row_acc+p_idx;
            let column_l_idx = 3*col_id;
            let column_t_idx = 3*(number_of_unique_points+(cam_idx-1));
            a.fixed_slice_mut::<1,3>(r_idx,column_l_idx).copy_from(&v_p_x);
            a.fixed_slice_mut::<1,3>(r_idx,column_t_idx).copy_from(&v_t_x);
            b.fixed_slice_mut::<1,3>(r_idx,column_l_idx).copy_from(&v_p_y);
            b.fixed_slice_mut::<1,3>(r_idx,column_t_idx).copy_from(&v_t_y);
            c.fixed_slice_mut::<1,3>(r_idx,column_l_idx).copy_from(&v_p_z);
            c.fixed_slice_mut::<1,3>(r_idx,column_t_idx).copy_from(&v_t_z);
        }

        row_acc += number_of_points; 
    }

    (a,b,c,a0,b0,c0)
}