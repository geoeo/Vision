extern crate nalgebra as na;
extern crate nalgebra_lapack;

use na::{Matrix2,Matrix3,Matrix4, OMatrix, Matrix3xX, SVector, Dynamic, dimension::{U10,U20,U9,U3}};
use crate::{Float,float};
use crate::sensors::camera::Camera;
use crate::image::{features::{Feature,Match},epipolar::{Essential,decompose_essential_förstner},triangulation::{linear_triangulation_svd,stereo_triangulation}};
use crate::numerics::{to_matrix, pose};

mod constraints;

/**
 * Photogrammetric Computer Vision p.575; Recent Developments on Direct Relative Orientation Nister, Stewenius et al.
 * Points may be planar
 * This only work on ubuntu. assert build version or something
 */
#[allow(non_snake_case)]
pub fn five_point_essential<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, camera_one: &C, camera_two: &C, image_dim: usize) -> Option<Essential> {
    let inverse_projection_one = camera_one.get_inverse_projection();
    let inverse_projection_two = camera_two.get_inverse_projection();
    let l = matches.len();
    let l_as_float = l as Float;
    
    let mut camera_rays_one = Matrix3xX::<Float>::zeros(l);
    let mut camera_rays_two = Matrix3xX::<Float>::zeros(l);
    let mut features_one = Matrix3xX::<Float>::zeros(l);
    let mut features_two = Matrix3xX::<Float>::zeros(l);
    let mut A = OMatrix::<Float, Dynamic,U9>::zeros(l);

    let mut normalization_matrix_one = Matrix3::<Float>::identity();
    let mut normalization_matrix_two = Matrix3::<Float>::identity();

    let mut avg_x_one = 0.0;
    let mut avg_y_one = 0.0;
    let mut max_dist_one: Float = 0.0;
    let mut avg_x_two = 0.0;
    let mut avg_y_two = 0.0;
    let mut max_dist_two: Float = 0.0;

    for i in 0..l {
        let m = &matches[i];
        let f_1_reduced = m.feature_one.get_camera_ray(&inverse_projection_one);
        let f_2_reduced = m.feature_two.get_camera_ray(&inverse_projection_two);

        camera_rays_one.column_mut(i).copy_from(&f_1_reduced);
        camera_rays_two.column_mut(i).copy_from(&f_2_reduced);

        let f_1 = m.feature_one.get_as_3d_point(camera_one.get_focal_x());
        let f_2 = m.feature_two.get_as_3d_point(camera_two.get_focal_x());
        avg_x_one += f_1[0];
        avg_y_one += f_1[1];
        max_dist_one = max_dist_one.max(f_1[0].powi(2) + f_1[1].powi(2));
        avg_x_two += f_2[0];
        avg_y_two += f_2[1];
        max_dist_two = max_dist_two.max(f_2[0].powi(2) + f_2[1].powi(2));

        features_one.column_mut(i).copy_from(&f_1);
        features_two.column_mut(i).copy_from(&f_2);
    }

    //TODO: unify with five_point and epipolar
    // normalization_matrix_one[(0,2)] = -avg_x_one/l_as_float;
    // normalization_matrix_one[(1,2)] = -avg_y_one/l_as_float;
    normalization_matrix_one[(2,2)] = max_dist_one;

    // normalization_matrix_two[(0,2)] = -avg_x_two/l_as_float;
    // normalization_matrix_two[(1,2)] = -avg_y_two/l_as_float;
    normalization_matrix_two[(2,2)] = max_dist_two;

    for i in 0..l {
        let c_x_1 = &camera_rays_one.column(i);
        let c_x_2 = &camera_rays_two.column(i);

        let kroenecker_product = c_x_2.kronecker(&c_x_1).transpose();
        A.row_mut(i).copy_from(&kroenecker_product);
    }

    let vt = match l {
        l if l < 5 => panic!("Five Point: Less than 5 features given!"),
        5 => nalgebra_lapack::SVD::new(A).expect("Five Point: SVD failed on A!").vt,
        _ => nalgebra_lapack::SVD::new(A.transpose()*A).expect("Five Point: SVD failed on A!").vt
    };

    let u1 = vt.row(5).transpose(); 
    let u2 = vt.row(6).transpose();
    let u3 = vt.row(7).transpose();
    let u4 = vt.row(8).transpose();

    let E1 = to_matrix::<_,3,3,9>(&u1).transpose();
    let E2 = to_matrix::<_,3,3,9>(&u2).transpose();
    let E3 = to_matrix::<_,3,3,9>(&u3).transpose();
    let E4 = to_matrix::<_,3,3,9>(&u4).transpose();

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

    let (eigenvalues, _, option_vr) = nalgebra_lapack::Eigen::complex_eigenvalues(action_matrix, false, true);
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
        let x = vec[6]/vec[9];
        let y = vec[7]/vec[9];
        let z = vec[8]/vec[9];

        let E_est = x*E1+y*E2+z*E3+E4;

        E_est
    }).collect::<Vec<Essential>>();
    let best_essential = cheirality_check(&all_essential_matricies, matches,false, image_dim,(&features_one, camera_one, &normalization_matrix_one), (&features_two, camera_two, &normalization_matrix_two));
    
    best_essential
}

#[allow(non_snake_case)]
pub fn cheirality_check<T: Feature + Clone,  C: Camera<Float>>(
        all_essential_matricies: &Vec<Essential>,
        matches: &Vec<Match<T>>,
        depth_positive: bool,
        image_dim: usize,
         points_cam_1: (&OMatrix<Float, U3,Dynamic>, &C, &Matrix3<Float>), 
         points_cam_2: (&OMatrix<Float, U3,Dynamic>, &C, &Matrix3<Float>)) -> Option<Essential> {
    let mut max_accepted_cheirality_count = 0;
    let mut best_e = None;
    let mut smallest_det = float::MAX;
    let camera_1 = points_cam_1.1;
    let camera_2 = points_cam_2.1;
    let camera_matrix_1 = camera_1.get_projection();
    let camera_matrix_2 = camera_2.get_projection();
    let condition_matrix_1 = points_cam_1.2; 
    let condition_matrix_2 = points_cam_2.2; 

    let number_of_points = matches.len();
    for e in all_essential_matricies {
        let (t,R,e_corrected) = decompose_essential_förstner(&e,matches,camera_1,camera_2);
        let se3 = pose::se3(&t,&R);

        let projection_1 = camera_matrix_1*(Matrix4::<Float>::identity().fixed_slice::<3,4>(0,0));
        let projection_2 = camera_matrix_2*(se3.fixed_slice::<3,4>(0,0));

        let p1_points = condition_matrix_1*points_cam_1.0;
        let p2_points = condition_matrix_2*points_cam_2.0;

        //TODO: review this with the sign change with better synthetic data
        //let Xs_option = Some(linear_triangulation_svd(&vec!((&p1_points,&projection_1),(&p2_points,&projection_2))));
        let Xs_option = stereo_triangulation((&p1_points,&projection_1),(&p2_points,&projection_2),image_dim as Float);
        match Xs_option {
            Some(Xs) => {
                let p1_x = projection_1*&Xs;
                let p2_x = projection_2*&Xs;
                let mut accepted_cheirality_count = 0;
                for i in 0..number_of_points {
                    let d1 = p1_x[(2,i)];
                    let d2 = p2_x[(2,i)];
        
                    if (depth_positive && d1 > 0.0 && d2 > 0.0) || (!depth_positive && d1 < 0.0 && d2 < 0.0) {
                        accepted_cheirality_count += 1 
                    }
                }
                let det = e_corrected.determinant().abs();
        
                let factor = e_corrected[(2,2)];
                let e_corrected_norm = e_corrected.map(|x| x/factor);
                // println!("{}",e_corrected);
                // println!("{}",e_corrected);
                // println!("{}",accepted_cheirality_count);
                // println!("{}",det);
                // println!("{}",se3);
                // println!("------");
        
                if (accepted_cheirality_count > max_accepted_cheirality_count) ||
                    ((accepted_cheirality_count == max_accepted_cheirality_count) && det < smallest_det) {
                    best_e = Some(e_corrected.clone());
                    smallest_det = det;
                    max_accepted_cheirality_count = accepted_cheirality_count;
                }
            },
            _=> ()
        };
    }
    // println!("------");
    best_e
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



