extern crate nalgebra as na;
extern crate nalgebra_lapack;

use na::{RowOVector,Vector3,Matrix2,Matrix3, Matrix4,Matrix3x4,OMatrix,Matrix3xX,SVector,Rotation3, dimension::{U10,U20,U5,U9,U3}};
use crate::{Float,float};
use crate::sensors::camera::Camera;
use crate::image::{features::{Feature,Match},epipolar::{Essential,decompose_essential_förstner},triangulation::linear_triangulation};
use crate::numerics::{to_matrix, pose,pose::{optimal_correction_of_rotation}};


mod constraints;

/**
 * Photogrammetric Computer Vision p.575; Recent Developments on Direct Relative Orientation Nister, Stewenius et al.
 * Points may be planar
 * This only work on ubuntu. assert build version or something
 * TODO: check with negative depth points and real data
 */
#[allow(non_snake_case)]
pub fn five_point_essential<T: Feature + Clone, C: Camera>(matches: &[Match<T>; 5], camera_one: &C, camera_two: &C, depth_positive: bool) -> Essential {
    let mut features_one = OMatrix::<Float, U3,U5>::zeros();
    let mut features_two = OMatrix::<Float, U3,U5>::zeros();
    let mut A = OMatrix::<Float,U5,U9>::zeros();
    let normalized_depth = match depth_positive {
        true => 1.0,
        _ => -1.0
    };

    let inverse_projection_one = camera_one.get_inverse_projection();
    let inverse_projection_two = camera_two.get_inverse_projection();

    for i in 0..5 {
        let m = &matches[i];
        let f_1 = m.feature_one.get_as_3d_point(normalized_depth);
        let f_2 = m.feature_two.get_as_3d_point(normalized_depth);

        features_one.column_mut(i).copy_from(&f_1);
        features_two.column_mut(i).copy_from(&f_2);
    }

    let camera_rays_one = inverse_projection_one*features_one;
    let camera_rays_two = inverse_projection_two*features_two;

    for i in 0..5 {
        let c_x_1 = &camera_rays_one.column(i);
        let c_x_2 = &camera_rays_two.column(i);

        let kroenecker_product = c_x_2.kronecker(&c_x_1).transpose();
        A.row_mut(i).copy_from(&kroenecker_product);
    }

    // nalgebra wont do full SVD
    let A_svd = nalgebra_lapack::SVD::new(A);
    let vt = &A_svd.expect("Five Point: SVD failed on A!").vt;
    let u1 = vt.row(5).transpose(); 
    let u2 = vt.row(6).transpose();
    let u3 = vt.row(7).transpose();
    let u4 = vt.row(8).transpose();

    let E1 = to_matrix::<3,3,9>(&u1).transpose();
    let E2 = to_matrix::<3,3,9>(&u2).transpose();
    let E3 = to_matrix::<3,3,9>(&u3).transpose();
    let E4 = to_matrix::<3,3,9>(&u4).transpose();

    let M = generate_five_point_constrait_matrix(&E1,&E2,&E3,&E4);

    let C = M.fixed_columns::<10>(0);
    let D = M.fixed_columns::<10>(10);
    
    let B = -C.try_inverse().expect("Five Point: Inverse of C failed!")*D;

    let mut action_matrix = OMatrix::<Float,U10,U10>::zeros();
    action_matrix.fixed_rows_mut::<3>(0).copy_from(&B.fixed_rows::<3>(0));
    action_matrix.fixed_rows_mut::<1>(3).copy_from(&B.fixed_rows::<1>(4));
    action_matrix.fixed_rows_mut::<1>(4).copy_from(&B.fixed_rows::<1>(5));
    action_matrix.fixed_rows_mut::<1>(5).copy_from(&B.fixed_rows::<1>(7));
    action_matrix.fixed_slice_mut::<2,2>(6,0).copy_from(&Matrix2::<Float>::identity());
    action_matrix[(8,3)] = 1.0;
    action_matrix[(9,6)] = 1.0;

    let (eigenvalues, option_vl, option_vr) = nalgebra_lapack::Eigen::complex_eigenvalues(action_matrix, false, true);
    let eigen_v = option_vr.expect("Five Point: eigenvector computation failed!");

    let mut real_eigenvalues =  Vec::<Float>::with_capacity(10);
    let mut real_eigenvectors = Vec::<SVector::<Float,10>>::with_capacity(10);
    for i in 0..10 {
        let c = eigenvalues[i];
        if c.im == 0.0 {
            let real_value = c.re;
            real_eigenvalues.push(real_value);
            real_eigenvectors.push(eigen_v.column(i).into_owned())
        }
    }

    let all_essential_matricies = real_eigenvectors.iter().map(|vec| {
        let u = vec[6];
        let v = vec[7];
        let w = vec[8];
        let t = vec[9];

        let x = vec[6]/vec[9];
        let y = vec[7]/vec[9];
        let z = vec[8]/vec[9];

        let E_est = x*E1+y*E2+z*E3+E4;
        E_est
    }).collect::<Vec<Essential>>();
    let matches_as_vec = matches.to_vec();
    let best_essential = cheirality_check(&all_essential_matricies, &matches_as_vec,depth_positive , (&camera_rays_one, &camera_one.get_projection(),&inverse_projection_one), (&camera_rays_two, &camera_two.get_projection(),&inverse_projection_two));
    
    best_essential
}

pub fn cheirality_check<T: Feature + Clone>(
        all_essential_matricies: &Vec<Essential>,
        matches_as_vec: &Vec<Match<T>>,
        depth_positive: bool,
         points_cam_1: (&OMatrix<Float, U3,U5>, &Matrix3<Float>,&Matrix3<Float>), 
         points_cam_2: (&OMatrix<Float, U3,U5>, &Matrix3<Float>,&Matrix3<Float>)) -> Essential {
    let mut max_accepted_cheirality_count = 0;
    let mut best_e = None;
    let mut smallest_det = float::MAX;
    let camera_matrix_1 = points_cam_1.1;
    let camera_matrix_2 = points_cam_2.1;
    let inverse_camera_matrix_1 = points_cam_1.2;
    let inverse_camera_matrix_2 = points_cam_2.2;
    for e in all_essential_matricies {
        let (t,R,e_corrected) = decompose_essential_förstner(&e,matches_as_vec,inverse_camera_matrix_1,inverse_camera_matrix_2,depth_positive);
        let R_corr = optimal_correction_of_rotation(&R);
        let se3 = pose::se3(&t,&R_corr);

        let projection_1 = camera_matrix_1*(Matrix4::<Float>::identity().fixed_slice::<3,4>(0,0));
        let projection_2 = camera_matrix_2*(se3.fixed_slice::<3,4>(0,0));

        let p1_static = points_cam_1.0;
        let p2_static = points_cam_2.0;
        let mut p1_dynamic = Matrix3xX::<Float>::zeros(5);
        let mut p2_dynamic = Matrix3xX::<Float>::zeros(5);
        p1_dynamic.fixed_columns_mut::<5>(0).copy_from(&p1_static.columns(0,5));
        p2_dynamic.fixed_columns_mut::<5>(0).copy_from(&p2_static.columns(0,5));

        let Xs = linear_triangulation(&vec!((&p1_dynamic,&projection_1),(&p2_dynamic,&projection_2)));
        let p1_x = projection_1*&Xs;
        let p2_x = projection_2*&Xs;
        let mut accepted_cheirality_count = 0;
        for i in 0..5 {
            let d1 = p1_x[(2,i)];
            let d2 = p2_x[(2,i)];

            if depth_positive && d1 > 0.0 && d2 > 0.0 || !depth_positive && d1 < 0.0 && d2 < 0.0 {
                accepted_cheirality_count += 1 
            }
        }

        let det = e_corrected.determinant().abs();
        if (accepted_cheirality_count > max_accepted_cheirality_count) ||
            ((accepted_cheirality_count == max_accepted_cheirality_count) && det < smallest_det) {
            best_e = Some(e_corrected);
            smallest_det = det;
            max_accepted_cheirality_count = accepted_cheirality_count;
        }
    }
    best_e.expect("cheirality_check: no best essential matrix found!").clone()
}

#[allow(non_snake_case)]
pub fn generate_five_point_constrait_matrix(E1: &Matrix3<Float>, E2: &Matrix3<Float>, E3: &Matrix3<Float>, E4: &Matrix3<Float>) -> OMatrix<Float,U10,U20> {

    let c_det_coeffs = constraints::get_determinant_constraints_coeffs(E1, E2, E3, E4);
    let c_1_coeffs = constraints::get_c1_constraints_coeffs(E1, E2, E3, E4);
    let c_2_coeffs = constraints::get_c2_constraints_coeffs(E1, E2, E3, E4);
    let c_3_coeffs = constraints::get_c3_constraints_coeffs(E1, E2, E3, E4);
    let c_4_coeffs = constraints::get_c4_constraints_coeffs(E1, E2, E3, E4);
    let c_5_coeffs = constraints::get_c5_constraints_coeffs(E1, E2, E3, E4);
    let c_6_coeffs = constraints::get_c6_constraints_coeffs(E1, E2, E3, E4);
    let c_7_coeffs = constraints::get_c7_constraints_coeffs(E1, E2, E3, E4);
    let c_8_coeffs = constraints::get_c8_constraints_coeffs(E1, E2, E3, E4);
    let c_9_coeffs = constraints::get_c9_constraints_coeffs(E1, E2, E3, E4);

    OMatrix::<Float,U10,U20>::from_rows(
        &[
        c_det_coeffs,
        c_1_coeffs,
        c_2_coeffs,
        c_3_coeffs,
        c_4_coeffs,
        c_5_coeffs,
        c_6_coeffs,
        c_7_coeffs,
        c_8_coeffs,
        c_9_coeffs
        ])
}



