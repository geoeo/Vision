extern crate nalgebra as na;

use std::collections::{HashMap,HashSet};
use rand::{rngs::SmallRng, SeedableRng, Rng};
use na::{Matrix4,Matrix3,Vector3, SMatrix,DVector, SVector,MatrixXx3,Matrix4xX,MatrixXx4,Matrix2xX,OMatrix,RowOVector,U3,U4};
use crate::image::features::{matches::Match,Feature};
use crate::sensors::camera::Camera;
use crate::Float;

#[derive(Clone, Copy)]
pub enum Triangulation {
    LINEAR,
    STEREO,
    LOST
}

/**
 * For a path pair (1,2) triangulates the feature in coordinate system of 1
 */
pub fn triangulate_matches<Feat: Feature, C: Camera<Float>>(path_pair: (usize, usize), pose: &Matrix4<Float>, 
    matches: & Vec<Match<Feat>>, camera_map: &HashMap<usize, C>, triangulation_mode: Triangulation) 
    -> Matrix4xX<Float> {
    let mut image_points_s = Matrix2xX::<Float>::zeros(matches.len());
    let mut image_points_f = Matrix2xX::<Float>::zeros(matches.len());
    
    // The position of the match in its vector indexes the value in the matrix
    for i in 0..matches.len() {
        let m = &matches[i];
        let feat_s = m.get_feature_one().get_as_2d_point();
        let feat_f = m.get_feature_two().get_as_2d_point();
        image_points_s.column_mut(i).copy_from(&feat_s);
        image_points_f.column_mut(i).copy_from(&feat_f);
    }
    
    let cam_1 = camera_map.get(&path_pair.0).expect("triangulate_matches: camera 1 not found");
    let cam_2 = camera_map.get(&path_pair.1).expect("triangulate_matches: camera 2 not found");

    let c1_intrinsics = cam_1.get_projection();
    let c2_intrinsics = cam_2.get_projection();
    let c1_inverse_intrinsics = cam_1.get_inverse_projection();
    let c2_inverse_intrinsics = cam_2.get_inverse_projection();
    let transform_c1 = Matrix4::<Float>::identity().fixed_view::<3,4>(0,0).into_owned();
    let transform_c2 = pose.fixed_view::<3,4>(0,0).into_owned();
    let projection_1 = c1_intrinsics*transform_c1;
    let projection_2 = c2_intrinsics*transform_c2;

    let f0 = 1.0;
    let f0_prime = 1.0;
    let pixel_error = 5.0; //TODO: configure this

    //TODO: Try to streamline > 2 views for applicable methods
    let landmarks = match triangulation_mode {
        Triangulation::LINEAR => linear_triangulation_svd(&vec!((&image_points_s,&projection_1),(&image_points_f,&projection_2)), true),
        Triangulation::STEREO => stereo_triangulation((&image_points_s,&projection_1),(&image_points_f,&projection_2),f0,f0_prime, true).expect("get_euclidean_landmark_state: Stereo Triangulation Failed"),
        Triangulation::LOST => linear_triangulation_lost(&vec!((&image_points_s,&c1_inverse_intrinsics,&Matrix4::<Float>::identity()),(&image_points_f,&c2_inverse_intrinsics,pose)), pixel_error)
    };

    landmarks
}

//TODO: maybe split up sign change
/**
 * Linear Triangulartion up to scale. Assuming norm(X) = 1, where X is in homogeneous space.
 * See Triangulation by Hartley et al.
 */
#[allow(non_snake_case)]
pub fn linear_triangulation_svd(image_points_and_projections: &Vec<(&Matrix2xX<Float>, &OMatrix<Float,U3,U4>)>, flip_points: bool) -> Matrix4xX<Float> {
    let n_cams = image_points_and_projections.len();
    let n_points = image_points_and_projections.first().expect("linear_triangulation: no points!").0.ncols();
    let mut triangulated_points = Matrix4xX::<Float>::zeros(n_points);

    for i in 0..n_points {
        let mut A = MatrixXx4::<Float>::zeros(2*n_cams);
        for j in 0..n_cams {
            let (points, projection) = image_points_and_projections[j];
            let p_1_1 = projection[(0,0)];
            let p_1_2 = projection[(0,1)];
            let p_1_3 = projection[(0,2)];
            let p_1_4 = projection[(0,3)];
    
            let p_2_1 = projection[(1,0)];
            let p_2_2 = projection[(1,1)];
            let p_2_3 = projection[(1,2)];
            let p_2_4 = projection[(1,3)];
    
            let p_3_1 = projection[(2,0)];
            let p_3_2 = projection[(2,1)];
            let p_3_3 = projection[(2,2)];
            let p_3_4 = projection[(2,3)];
            let u = points[(0,i)];
            let v = points[(1,i)];
            A.fixed_rows_mut::<1>(2*j).copy_from(&RowOVector::<Float,U4>::from_vec(vec![u*p_3_1 - p_1_1, u*p_3_2-p_1_2, u*p_3_3-p_1_3, u*p_3_4-p_1_4]));
            A.fixed_rows_mut::<1>(2*j+1).copy_from(&RowOVector::<Float,U4>::from_vec(vec![v*p_3_1 - p_2_1, v*p_3_2-p_2_2, v*p_3_3-p_2_3, v*p_3_4-p_2_4]));
        }


        let svd = (A.transpose()*A).svd(false,true);
        let eigen_vectors = svd.v_t.expect("linear_triangulation: svd failed");

        let p = eigen_vectors.row((2*n_cams)-1);

        triangulated_points[(0,i)] = p[0]/p[3];
        triangulated_points[(1,i)] = p[1]/p[3];
        triangulated_points[(2,i)] = p[2]/p[3];
        triangulated_points[(3,i)] = 1.0;

        if flip_points {
            // We may triangulate points begind the camera. So we flip them depending on the principal distance
            let sign = match triangulated_points[(2,i)].is_sign_negative() {
                true => -1.0,
                false => 1.0
            };

            triangulated_points[(0,i)] *= sign;
            triangulated_points[(1,i)] *= sign;
            triangulated_points[(2,i)] *= sign;
        }
    }
    triangulated_points
}

/**
 * Stereo Triangulation. Only works for two-view problem but optimizes based on the epipolar constraints.
 * See 3D Rotations - Kanatani
 */
#[allow(non_snake_case)]
pub fn stereo_triangulation(image_points_and_projection: (&Matrix2xX<Float>, &OMatrix<Float,U3,U4>), image_points_and_projection_prime: (&Matrix2xX<Float>, &OMatrix<Float,U3,U4>), f0: Float, f0_prime: Float, flip_points: bool) -> Option<Matrix4xX<Float>> {
    let (image_points, projection) =  image_points_and_projection;
    let (image_points_prime, projection_prime) =  image_points_and_projection_prime;

    assert_eq!(image_points.ncols(),image_points_prime.ncols());
    let n = image_points.ncols();
    let mut triangulated_points = Matrix4xX::<Float>::zeros(n);
    let mut success = true;

    for i in 0..n {
        let im_point = image_points.column(i);
        let im_point_prime = image_points_prime.column(i);
        let x = im_point[0];
        let y = im_point[1];
        let x_prime = im_point_prime[0];
        let y_prime = im_point_prime[1];

        let P_11 = projection[(0,0)];
        let P_12 = projection[(0,1)];
        let P_13 = projection[(0,2)];
        let P_14 = projection[(0,3)];

        let P_21 = projection[(1,0)];
        let P_22 = projection[(1,1)];
        let P_23 = projection[(1,2)];
        let P_24 = projection[(1,3)];

        let P_31 = projection[(2,0)];
        let P_32 = projection[(2,1)];
        let P_33 = projection[(2,2)];
        let P_34 = projection[(2,3)];

        let P_11_prime = projection_prime[(0,0)];
        let P_12_prime = projection_prime[(0,1)];
        let P_13_prime = projection_prime[(0,2)];
        let P_14_prime = projection_prime[(0,3)];

        let P_21_prime = projection_prime[(1,0)];
        let P_22_prime = projection_prime[(1,1)];
        let P_23_prime = projection_prime[(1,2)];
        let P_24_prime = projection_prime[(1,3)];

        let P_31_prime = projection_prime[(2,0)];
        let P_32_prime = projection_prime[(2,1)];
        let P_33_prime = projection_prime[(2,2)];
        let P_34_prime = projection_prime[(2,3)];

        let T = SMatrix::<Float,4,3>::new(
            f0*P_11-x*P_31,f0*P_12-x*P_32,f0*P_13-x*P_33,
            f0*P_21-y*P_31,f0*P_22-y*P_32,f0*P_23-y*P_33,
            f0_prime*P_11_prime-x_prime*P_31_prime,f0_prime*P_12_prime-x_prime*P_32_prime,f0_prime*P_13_prime-x_prime*P_33_prime,
            f0_prime*P_21_prime-y_prime*P_31_prime,f0_prime*P_22_prime-y_prime*P_32_prime,f0_prime*P_23_prime-y_prime*P_33_prime
        );
        let p = SVector::<Float,4>::new(
            f0*P_14-x*P_34,
            f0*P_24-y*P_34,
            f0_prime*P_14_prime-x*P_34_prime,
            f0_prime*P_24_prime-y*P_34_prime
        );
        let T_transpose = T.transpose();
        let b = T_transpose*p;
        
        match (T_transpose*T).qr().solve(&b) {
            Some(x) => {
                triangulated_points[(0,i)] = x[(0,0)];
                triangulated_points[(1,i)] = x[(1,0)];
                triangulated_points[(2,i)] = x[(2,0)];
                triangulated_points[(3,i)] = 1.0;

                if flip_points {
                    // We may triangulate points begind the camera. So we flip them depending on the principal distance
                    let sign = match triangulated_points[(2,i)].is_sign_negative() {
                        true => -1.0,
                        false => 1.0
                    };
        
                    triangulated_points[(0,i)] *= sign;
                    triangulated_points[(1,i)] *= sign;
                    triangulated_points[(2,i)] *= sign;
                }

            },
            _ => {success = false;}
        };


    }

    match success {
        true => Some(triangulated_points),
        false => None
    }
    
}

/**
 * LOST Triangulation 
 * See https://gtsam.org/2023/02/04/lost-triangulation.html / https://arxiv.org/pdf/2205.12197.pdf
 */
#[allow(non_snake_case)]
pub fn linear_triangulation_lost(image_points_and_projections: &Vec<(&Matrix2xX<Float>, &Matrix3<Float>, &Matrix4<Float>)>, pixel_error: Float) -> Matrix4xX<Float> {
    let mut sampling = rand::rngs::SmallRng::from_entropy();

    let n_cams = image_points_and_projections.len();
    let n_points = image_points_and_projections.first().expect("linear_triangulation: no points!").0.ncols();
    let mut triangulated_points = Matrix4xX::<Float>::zeros(n_points);
    let camera_indices = (0..n_cams).collect::<HashSet<_>>();
    
    for i in 0..n_points {
        let mut A = MatrixXx3::<Float>::zeros(2*n_cams);
        let mut b = DVector::<Float>::zeros(2*n_cams);
        for j in 0..n_cams {
            let companion_idx = pick_companion_camera(j,&camera_indices,&mut sampling);
            // The transform transforms a point from coordiante system i to reference cam, which is 0 in our case
            let (points, inverse_intrinsics, transform_zero_i) = image_points_and_projections[j];
            let (points_companion, inverse_intrinsics_companion, transform_zero_i_companion) = image_points_and_projections[companion_idx];

            let u = points[(0,i)];
            let v = points[(1,i)];
            let u_companion = points_companion[(0,i)];
            let v_companion = points_companion[(1,i)];
            let pixel_pos = Vector3::<Float>::new(u,v,1.0);
            let pixel_pos_companion = Vector3::<Float>::new(u_companion,v_companion,1.0);
            let rotation = transform_zero_i.fixed_view::<3,3>(0,0);
            let q_factor = compute_q_factor(pixel_error, inverse_intrinsics, inverse_intrinsics_companion, transform_zero_i, transform_zero_i_companion, &pixel_pos, &pixel_pos_companion);

            let elem_cross = (inverse_intrinsics*pixel_pos).cross_matrix();
            let a_elem = q_factor*elem_cross*rotation.transpose();
            let b_elem_cross = a_elem*pixel_pos;

            A.fixed_view_mut::<2,3>(j, 0).copy_from(&a_elem.fixed_view::<2,3>(0,0));
            b.fixed_rows_mut::<2>(j).copy_from(&b_elem_cross.fixed_rows::<2>(0));

        }   

        let svd = A.svd(true, true);
        let landmark = svd.solve(&b,1e-6).expect("LOST LU Failed");

        //TODO: check if this is still neccessary!
        let sign = match landmark[2].is_sign_negative() {
            true => -1.0,
            false => 1.0
        };

        triangulated_points[(0,i)] = landmark[0]*sign;
        triangulated_points[(1,i)] = landmark[1]*sign;
        triangulated_points[(2,i)] = landmark[2]*sign;
        triangulated_points[(3,i)] = 1.0;
    }

    triangulated_points
}

fn pick_companion_camera(cam_index: usize, camera_indices: &HashSet<usize>, sampling: &mut SmallRng) -> usize {
    let reduced_indices = camera_indices.into_iter().filter(|&x| *x!= cam_index).collect::<Vec<&usize>>();
    let max = reduced_indices.len()-1;
    let random_idx = sampling.gen::<f32>()*(max as f32);
    *reduced_indices[random_idx.floor() as usize]
}

fn compute_q_factor(
    pixel_error: Float, 
    inverse_intrinsics: &Matrix3<Float>, 
    inverse_intrinsics_companion: &Matrix3<Float>, 
    transform: &Matrix4<Float>,
    transform_companion: &Matrix4<Float>,
    pixel_pos:&Vector3<Float>,
    pixel_pos_companion:&Vector3<Float>) -> Float {
        let x_sigma = pixel_error * inverse_intrinsics[(0,0)];
        let rotation = transform.fixed_view::<3,3>(0,0);
        let position = transform.fixed_view::<3,1>(0,3);
        let rotation_companion = transform_companion.fixed_view::<3,3>(0,0);
        let position_companion = transform_companion.fixed_view::<3,1>(0,3);
        let baseline = position_companion - position;

        let numerator_left = rotation*inverse_intrinsics*pixel_pos;
        let numerator_right = rotation_companion*inverse_intrinsics_companion*pixel_pos_companion;
        let numerator = numerator_left.cross(&numerator_right).norm();
        let denominator = x_sigma*baseline.cross(&numerator_right).norm();

        numerator/denominator
}