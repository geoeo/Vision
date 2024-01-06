extern crate nalgebra as na;
extern crate nalgebra_lapack;

use std::marker::{Sync,Send};
use na::{Matrix3,Matrix4, OMatrix ,Matrix3xX, SVector, Dyn, dimension::{U10,U20,U9,U3},linalg::SymmetricEigen};
use crate::{Float,float};
use crate::image::features::{Feature,matches::Match};
use crate::sfm::{triangulation::linear_triangulation_svd,epipolar::{Essential,tensor::decompose_essential_förstner}};
use crate::numerics::to_matrix;

mod constraints;

/**
 * Photogrammetric Computer Vision p.575; Recent Developments on Direct Relative Orientation Nister, Stewenius et al.
 * Points may be planar
 * This only work on ubuntu. assert build version or something
 */
#[allow(non_snake_case)]
pub fn five_point_essential<T: Feature + Clone+ Send + Sync>(matches: &Vec<Match<T>>, projection_one: &Matrix3<Float>, inverse_projection_one: &Matrix3<Float>, projection_two:&Matrix3<Float>,inverse_projection_two: &Matrix3<Float>) -> Option<Essential> {
    let l = matches.len();
    
    let mut camera_rays_one = Matrix3xX::<Float>::zeros(l);
    let mut camera_rays_two = Matrix3xX::<Float>::zeros(l);
    let mut features_one = Matrix3xX::<Float>::zeros(l);
    let mut features_two = Matrix3xX::<Float>::zeros(l);
    let mut A = OMatrix::<Float, Dyn,U9>::zeros(l);

    for i in 0..l {
        let m = &matches[i];
        let f_1_reduced = m.get_feature_one().get_camera_ray_photogrammetric(&inverse_projection_one);
        let f_2_reduced = m.get_feature_two().get_camera_ray_photogrammetric(&inverse_projection_two);

        camera_rays_one.column_mut(i).copy_from(&f_1_reduced);
        camera_rays_two.column_mut(i).copy_from(&f_2_reduced);

        let f_1 = m.get_feature_one().get_as_homogeneous(1.0);
        let f_2 = m.get_feature_two().get_as_homogeneous(1.0);

        features_one.column_mut(i).copy_from(&f_1);
        features_two.column_mut(i).copy_from(&f_2);
    }

    for i in 0..l {
        let c_x_1 = &camera_rays_one.column(i);
        let c_x_2 = &camera_rays_two.column(i);

        let kroenecker_product = c_x_2.kronecker(&c_x_1).transpose();
        A.row_mut(i).copy_from(&kroenecker_product);
    }

    let (u1, u2, u3, u4) = match l {
        l if l < 5 => panic!("Five Point: Less than 5 features given!"),
        5 => {
            let vt = nalgebra_lapack::SVD::new(A).expect("Five Point: SVD failed on A!").vt;
            let u1 = vt.row(5).transpose(); 
            let u2 = vt.row(6).transpose();
            let u3 = vt.row(7).transpose();
            let u4 = vt.row(8).transpose(); //smallest
            (u1, u2, u3, u4)
        },
        _ => {
            let A_hat = A.transpose()*&A;
            let eigen = SymmetricEigen::new(A_hat);
            let eigenvectors = eigen.eigenvectors;
            let mut indexed_eigenvalues = eigen.eigenvalues.iter().enumerate().map(|(i,v)| (i,*v)).collect::<Vec<(usize, Float)>>();
            indexed_eigenvalues.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let u1 = eigenvectors.column(indexed_eigenvalues[0].0).into_owned();
            let u2 = eigenvectors.column(indexed_eigenvalues[1].0).into_owned();
            let u3 = eigenvectors.column(indexed_eigenvalues[2].0).into_owned();
            let u4 = eigenvectors.column(indexed_eigenvalues[3].0).into_owned();
            (u4, u3, u2, u1)
        }
    };

    let E1 = to_matrix::<_,3,3,9>(&u1);
    let E2 = to_matrix::<_,3,3,9>(&u2);
    let E3 = to_matrix::<_,3,3,9>(&u3);
    let E4 = to_matrix::<_,3,3,9>(&u4);

    let M = generate_five_point_constrait_matrix(&E1,&E2,&E3,&E4);
    let C = M.fixed_columns::<10>(0);
    let D = M.fixed_columns::<10>(10);
    let C_inv_option = C.try_inverse();
    match C_inv_option {
        Some(C_inv) => {
            let B = -C_inv*D;
            let mut action_matrix = OMatrix::<Float,U10,U10>::zeros();
            action_matrix.fixed_rows_mut::<3>(0).copy_from(&B.fixed_rows::<3>(0));
            action_matrix.fixed_rows_mut::<1>(3).copy_from(&B.fixed_rows::<1>(4));
            action_matrix.fixed_rows_mut::<1>(4).copy_from(&B.fixed_rows::<1>(5));
            action_matrix.fixed_rows_mut::<1>(5).copy_from(&B.fixed_rows::<1>(7));
            action_matrix[(6,0)] = 1.0;
            action_matrix[(7,1)] = 1.0;
            action_matrix[(8,3)] = 1.0;
            action_matrix[(9,6)] = 1.0;
        
            let eigen = nalgebra_lapack::Eigen::new(action_matrix.transpose(), false, true).expect("Five Point: eigenvector computation failed!");
            let eigen_v = eigen.eigenvectors.expect("Five Point: could not retrieve right eigenvectors!");
        
            let mut real_eigenvectors = Vec::<SVector::<Float,10>>::with_capacity(10);
            for i in 0..10 {
                real_eigenvectors.push(eigen_v.column(i).into_owned());
            }
            
            let all_essential_matricies = real_eigenvectors.iter().map(|vec| {
                let x = vec[6]/vec[9];
                let y = vec[7]/vec[9];
                let z = vec[8]/vec[9];
        
                x*E1+y*E2+z*E3+E4
            }).collect::<Vec<Essential>>();
            let best_essential = cheirality_check(&all_essential_matricies, matches,(&features_one, projection_one, inverse_projection_one), (&features_two, projection_two,inverse_projection_two));
            
            best_essential
        },
        None => None
    }

}

#[allow(non_snake_case)]
pub fn cheirality_check<T: Feature + Clone+ Send + Sync>(
        all_essential_matricies: &Vec<Essential>,
        matches: &Vec<Match<T>>,
        points_cam_1: (&OMatrix<Float, U3,Dyn>, &Matrix3<Float>,&Matrix3<Float>), 
        points_cam_2: (&OMatrix<Float, U3,Dyn>, &Matrix3<Float>,&Matrix3<Float>)) -> Option<Essential> {
    let mut max_accepted_cheirality_count = 0;
    let mut best_e = None;
    let mut smallest_det = float::MAX;

    let camera_matrix_1 = points_cam_1.1;
    let camera_matrix_2 = points_cam_2.1;
    
    let number_of_points = matches.len();
    for e in all_essential_matricies {
        let (iso3_option,e_corrected) = decompose_essential_förstner(&e,matches,points_cam_1.2,points_cam_2.2);
        match iso3_option {
            Some(iso3) => {
                let se3 = iso3.to_matrix();
                let projection_1 = camera_matrix_1*(Matrix4::<Float>::identity().fixed_view::<3,4>(0,0));
                let projection_2 = camera_matrix_2*(se3.fixed_view::<3,4>(0,0));
        
                let p1_points = points_cam_1.0;
                let p2_points = points_cam_2.0;
        
                let Xs_option = Some(linear_triangulation_svd(&vec!((&p1_points,&projection_1),(&p2_points,&projection_2)), false));
        
                match Xs_option {
                    Some(Xs) => {
                        let p1_x = &Xs;
                        let p2_x = se3*&Xs;
        
                        let mut accepted_cheirality_count = 0;
                        for i in 0..number_of_points {
                            let d1 = p1_x[(2,i)];
                            let d2 = p2_x[(2,i)];
        
                            if d1 > 0.0 && d2 > 0.0 {
                                accepted_cheirality_count += 1 
                            }
                        }
                        let det = e_corrected.determinant().abs();
        
                        if  !det.is_nan() && ((accepted_cheirality_count >= max_accepted_cheirality_count) ||
                            ((accepted_cheirality_count == max_accepted_cheirality_count) && det < smallest_det)) {
                            best_e = Some(e_corrected.clone());
                            smallest_det = det;
                            max_accepted_cheirality_count = accepted_cheirality_count;
                        }
                    },
                    _=> ()
                };
            },
            None => ()
        };
    }
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



