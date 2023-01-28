extern crate nalgebra as na;
extern crate nalgebra_lapack;

use na::{Matrix3,Matrix4, OMatrix ,Matrix3xX, SVector, Dynamic, dimension::{U10,U20,U9,U3}};
use crate::{Float,float};
use crate::sensors::camera::Camera;
use crate::image::features::{Feature,Match};
use crate::sfm::{triangulation::{linear_triangulation_svd,stereo_triangulation},epipolar::{Essential,tensor::decompose_essential_förstner}};
use crate::numerics::{to_matrix, pose};

mod constraints;

/**
 * Photogrammetric Computer Vision p.575; Recent Developments on Direct Relative Orientation Nister, Stewenius et al.
 * Points may be planar
 * This only work on ubuntu. assert build version or something
 */
#[allow(non_snake_case)]
pub fn five_point_essential<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, camera_one: &C, camera_two: &C) -> Option<Essential> {
    let inverse_projection_one = camera_one.get_inverse_projection();
    let inverse_projection_two = camera_two.get_inverse_projection();
    let projection_one = camera_one.get_projection();
    let projection_two = camera_two.get_projection();

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
    let mut avg_x_two = 0.0;
    let mut avg_y_two = 0.0;

    let mut max_x_one: Float = 0.0;
    let mut max_y_one: Float = 0.0;
    let mut max_x_two: Float = 0.0;
    let mut max_y_two: Float = 0.0;

    for i in 0..l {
        let m = &matches[i];
        let f_1_reduced = m.feature_one.get_camera_ray(&inverse_projection_one);
        let f_2_reduced = m.feature_two.get_camera_ray(&inverse_projection_two);

        camera_rays_one.column_mut(i).copy_from(&f_1_reduced);
        camera_rays_two.column_mut(i).copy_from(&f_2_reduced);

        let f_1 = m.feature_one.get_as_3d_point(-1.0);
        let f_2 = m.feature_two.get_as_3d_point(-1.0);

        max_x_one = max_x_one.max(f_1[0]);
        max_y_one = max_y_one.max(f_1[1]);

        max_x_two = max_x_two.max(f_2[0]);
        max_y_two = max_y_two.max(f_2[1]);

        avg_x_one += f_1[0];
        avg_y_one += f_1[1];
        avg_x_two += f_2[0];
        avg_y_two += f_2[1];

        features_one.column_mut(i).copy_from(&f_1);
        features_two.column_mut(i).copy_from(&f_2);
    }

    // let cx_one = projection_one[(0,2)];
    // let cy_one = projection_one[(1,2)];
    // let cx_two = projection_two[(0,2)];
    // let cy_two = projection_two[(1,2)];

    // let max_dist_one = max_x_one*max_y_one;
    // let max_dist_two = max_x_two*max_y_two;

    // let max_dist_one = (cx_one.powi(2)+cy_one.powi(2)).sqrt();
    // let max_dist_two = (cx_two.powi(2)+cy_two.powi(2)).sqrt();

    // let max_dist_one = 1.0;
    // let max_dist_two = 1.0;


    //TODO: unify with five_point and epipolar
    // normalization_matrix_one[(0,2)] = -avg_x_one/(l_as_float);
    // normalization_matrix_one[(1,2)] = -avg_y_one/(l_as_float);
    // normalization_matrix_one[(2,2)] = max_dist_one;

    // normalization_matrix_two[(0,2)] = -avg_x_two/(l_as_float);
    // normalization_matrix_two[(1,2)] = -avg_y_two/(l_as_float);
    // normalization_matrix_two[(2,2)] = max_dist_two;

    for i in 0..l {
        let c_x_1 = &camera_rays_one.column(i);
        let c_x_2 = &camera_rays_two.column(i);

        let kroenecker_product = c_x_2.kronecker(&c_x_1).transpose();
        A.row_mut(i).copy_from(&kroenecker_product);
    }

    let (u1, u2, u3, u4) = match l {
        l if l < 5 => panic!("Five Point: Less than 5 features given!"),
        5 => {
            //TODO: check order for strickly 5 points
            let vt = nalgebra_lapack::SVD::new(A).expect("Five Point: SVD failed on A!").vt;
            let u1 = vt.row(5).transpose(); 
            let u2 = vt.row(6).transpose();
            let u3 = vt.row(7).transpose();
            let u4 = vt.row(8).transpose();
            (u1, u2, u3, u4)
        },
        _ => {
            let eigen = nalgebra_lapack::SymmetricEigen::new(A.transpose()*A);
            let eigenvectors = eigen.eigenvectors;
            let mut indexed_eigenvalues = eigen.eigenvalues.iter().enumerate().map(|(i,v)| (i,*v)).collect::<Vec<(usize, Float)>>();
            indexed_eigenvalues.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let u1 = eigenvectors.column(indexed_eigenvalues[0].0).into_owned();
            let u2 = eigenvectors.column(indexed_eigenvalues[1].0).into_owned();
            let u3 = eigenvectors.column(indexed_eigenvalues[2].0).into_owned();
            let u4 = eigenvectors.column(indexed_eigenvalues[3].0).into_owned();

            (u1, u2, u3, u4)
        }

    };


    let E1 = to_matrix::<_,3,3,9>(&u1);
    let E2 = to_matrix::<_,3,3,9>(&u2);
    let E3 = to_matrix::<_,3,3,9>(&u3);
    let E4 = to_matrix::<_,3,3,9>(&u4);

    let M = generate_five_point_constrait_matrix(&E1,&E2,&E3,&E4);

    let C = M.fixed_columns::<10>(0);
    let D = M.fixed_columns::<10>(10);
    
    let B = -C.try_inverse().expect("Five Point: Inverse of C failed!")*D;

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

        let E_est = x*E1+y*E2+z*E3+E4;
        E_est
    }).collect::<Vec<Essential>>();
    let best_essential = cheirality_check(&all_essential_matricies, matches,false,(&features_one, camera_one, &normalization_matrix_one), (&features_two, camera_two, &normalization_matrix_two));
    
    best_essential
}

#[allow(non_snake_case)]
pub fn cheirality_check<T: Feature + Clone,  C: Camera<Float>>(
        all_essential_matricies: &Vec<Essential>,
        matches: &Vec<Match<T>>,
        depth_positive: bool,
         points_cam_1: (&OMatrix<Float, U3,Dynamic>, &C, &Matrix3<Float>), 
         points_cam_2: (&OMatrix<Float, U3,Dynamic>, &C, &Matrix3<Float>)) -> Option<Essential> {
    let mut max_accepted_cheirality_count = 0;
    let mut best_e = None;
    let mut smallest_det = float::MAX;
    let camera_1 = points_cam_1.1;
    let camera_2 = points_cam_2.1;

    //TODO: clean this up
    let camera_matrix_1 = camera_1.get_projection();
    let camera_matrix_2 = camera_2.get_projection();
    let f0 = 1.0;
    let f0_prime = 1.0;
    // let condition_matrix_1 = points_cam_1.2; 
    // let condition_matrix_2 = points_cam_2.2; 
    // let f0 = condition_matrix_1[(2,2)];
    // let f0_prime = condition_matrix_2[(2,2)];

    // camera_matrix_1[(0,0)] /= f0;
    // camera_matrix_1[(1,1)] /= f0;
    // camera_matrix_1[(0,2)] /= f0;
    // camera_matrix_1[(1,2)] /= f0;

    // camera_matrix_2[(0,0)] /= f0_prime;
    // camera_matrix_2[(1,1)] /= f0_prime;
    // camera_matrix_2[(0,2)] /= f0_prime;
    // camera_matrix_2[(1,2)] /= f0_prime;


    let number_of_points = matches.len();
    for e in all_essential_matricies {
        let (t,R,e_corrected) = decompose_essential_förstner(&e,matches,&camera_1.get_inverse_projection(),&camera_2.get_inverse_projection());
        let se3 = pose::se3(&t,&R);

        let projection_1 = camera_matrix_1*(Matrix4::<Float>::identity().fixed_slice::<3,4>(0,0));
        let projection_2 = camera_matrix_2*(se3.fixed_slice::<3,4>(0,0));

        let p1_points = points_cam_1.0/f0;
        let p2_points = points_cam_2.0/f0_prime;

        //TODO make ENUM
        //let Xs_option = Some(linear_triangulation_svd(&vec!((&p1_points,&projection_1),(&p2_points,&projection_2))));
        let Xs_option = stereo_triangulation((&p1_points,&projection_1),(&p2_points,&projection_2),f0,f0_prime);
        match Xs_option {
            Some(Xs) => {
                let p1_x = &Xs;
                let p2_x = se3*&Xs;

                let mut accepted_cheirality_count = 0;
                for i in 0..number_of_points {
                    let d1 = p1_x[(2,i)];
                    let d2 = p2_x[(2,i)];
        
                    if (depth_positive && d1 > 0.0 && d2 > 0.0) || (!depth_positive && d1 < 0.0 && d2 < 0.0) {
                        accepted_cheirality_count += 1 
                    }
                }
                let e_corrected_norm = e_corrected.normalize();
                let det = e_corrected_norm.determinant().abs();

                if  !det.is_nan() && ((accepted_cheirality_count >= max_accepted_cheirality_count) ||
                    ((accepted_cheirality_count == max_accepted_cheirality_count) && det < smallest_det)) {
                    best_e = Some(e_corrected_norm.clone());
                    smallest_det = det;
                    max_accepted_cheirality_count = accepted_cheirality_count;
                }
            },
            _=> ()
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



