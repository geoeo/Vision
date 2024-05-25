extern crate nalgebra as na;

use na::{Matrix3,Matrix4,Matrix5,Matrix6,Vector4,Isometry3,Matrix3x4,Matrix4x6};
use std::collections::HashMap;
use crate::sfm::state::landmark::Landmark;
use crate::Float;
use crate::image::features::{matches::Match,Feature};
use crate::sensors::camera::Camera;
use crate::sfm::state::landmark::euclidean_landmark::EuclideanLandmark;

/**
 * Determining an initial image pair for fixing the
 * scale of a 3d reconstruction from an image
 * sequence. Beder et al.
 */
 #[allow(non_snake_case)]
fn estimate_covariance_for_match<Feat: Feature>(
        m: &Match<Feat>, 
        landmark_world: &Vector4<Float>, 
        extrinsics_one: &Isometry3<Float>,
        inverse_intrinsics_one: &Matrix3<Float>,
        extrinsics_two: &Isometry3<Float>,
        inverse_intrinsics_two: &Matrix3<Float>,
        pixel_std: Float
    ) -> Matrix3<Float> {
        let pixel_covariance = Matrix3::<Float>::from_diagonal_element(pixel_std.powi(2));
        let ray_one_covariance = inverse_intrinsics_one*pixel_covariance*inverse_intrinsics_one.transpose();
        let ray_two_covariance = inverse_intrinsics_two*pixel_covariance*inverse_intrinsics_two.transpose();

        let ray_one = m.get_feature_one().get_camera_ray(inverse_intrinsics_one);
        let ray_two = m.get_feature_two().get_camera_ray(inverse_intrinsics_two);
        
        let landmark_cam = extrinsics_one.inverse().to_matrix()*landmark_world;
        // projective matricies with the intrinsic component as identity - origin at cam1
        let p_one = (extrinsics_one.inverse()*extrinsics_one).to_matrix().fixed_view::<3,4>(0,0).into_owned();
        let p_two = (extrinsics_one.inverse()*extrinsics_two).to_matrix().fixed_view::<3,4>(0,0).into_owned();

        let a_r12 = ray_one.cross_matrix().fixed_rows::<2>(0)*p_one;
        let a_r34 = ray_two.cross_matrix().fixed_rows::<2>(0)*p_two;
        let mut A = Matrix4::<Float>::zeros();
        A.fixed_view_mut::<2,4>(0,0).copy_from(&a_r12);
        A.fixed_view_mut::<2,4>(2,0).copy_from(&a_r34);

        let b_upper =  -(p_one*landmark_cam).cross_matrix().fixed_rows::<2>(0);
        let b_lower =  -(p_two*landmark_cam).cross_matrix().fixed_rows::<2>(0);
        let mut B = Matrix4x6::<Float>::zeros();
        B.fixed_view_mut::<2,3>(0,0).copy_from(&b_upper);
        B.fixed_view_mut::<2,3>(2,3).copy_from(&b_lower);

        let mut cov_temp = Matrix6::<Float>::zeros();
        cov_temp.fixed_view_mut::<3,3>(0,0).copy_from(&ray_one_covariance);
        cov_temp.fixed_view_mut::<3,3>(3,3).copy_from(&ray_two_covariance);
        let n_ul = A.transpose()*(B*cov_temp*B.transpose()).try_inverse().expect("estimate_covariance_for_match: Inverse of N upper failed!");
        let n_ur = A*landmark_cam;

        let mut N = Matrix5::<Float>::zeros();
        N.fixed_view_mut::<4,4>(0,0).copy_from(&n_ul);
        N.fixed_view_mut::<4,1>(0,4).copy_from(&n_ur);
        N.fixed_view_mut::<1,4>(4,0).copy_from(&landmark_cam.transpose());

        let n_inv = N.try_inverse().expect("estimate_covariance_for_match: Inverse of N failed!");
        let cov_point = n_inv.fixed_view::<4,4>(0,0);
        let j_e = Matrix3x4::<Float>::new(
            1.0/landmark_cam.w,0.0,0.0,-landmark_cam.x/landmark_cam.w.powi(2),
            0.0,1.0/landmark_cam.w,0.0,-landmark_cam.y/landmark_cam.w.powi(2),
            0.0,0.0,1.0/landmark_cam.w,-landmark_cam.z/landmark_cam.w.powi(2)
        );
        j_e*cov_point*j_e.transpose()
}

/**
 * Score covariance by roundness factor. Best value for reconstruction is a score of sqrt(1/2)
 */
fn score_covariance(cov: &Matrix3<Float>) -> Float {
    let svd = cov.svd(false, false);
    (svd.singular_values[2] / svd.singular_values[0]).sqrt()
}

pub fn score_camera_pairs<Feat: Feature, C: Camera<Float>>(
    match_map:  &HashMap<(usize, usize), Vec<Match<Feat>>>, 
    cam_map: &HashMap<usize, C>,
    landmark_map: &HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>,
    abs_pose_map: &HashMap<usize, Isometry3<Float>>,
    pixel_std: Float
) -> HashMap<(usize,usize),Float> {
    match_map.iter().map(|((cam_id_1, cam_id_2),ms)| {
        let cam_1 = cam_map.get(cam_id_1).expect("score_camera_pair: Could not find cam");
        let cam_2 = cam_map.get(cam_id_2).expect("score_camera_pair: Could not find cam");
        let landmarks = landmark_map.get(&(*cam_id_1, *cam_id_2)).expect("score_camera_pair: Could not find landmarks");
        let p1 = abs_pose_map.get(cam_id_1).expect("score_camera_pair: could not find pose");
        let p2 = abs_pose_map.get(cam_id_2).expect("score_camera_pair: could not find pose");
        
        // As per the paper we remap poses such that cam 1 is the origin
        let p1_prime = p1.inverse()*p1;
        let p2_prime = p1.inverse()*p2;
        let inv_intrinsics_1 = cam_1.get_inverse_projection();
        let inv_intrinsics_2 = cam_2.get_inverse_projection();

        let mut score_acc: Float = 0.0;
        for i in 0..landmarks.len() {
            let m = &ms[i];
            let l_vec = landmarks[i].get_state_as_vector();
            let l_vec4 = Vector4::<Float>::new(l_vec.x,l_vec.y,l_vec.z,1.0);
            let cov = estimate_covariance_for_match(m,&l_vec4,&p1_prime,&inv_intrinsics_1,&p2_prime,&inv_intrinsics_2,pixel_std);
            let score_for_m = score_covariance(&cov);
            let diff_to_target = ((0.5 as Float).sqrt() - score_for_m).abs();
            score_acc += diff_to_target;
        }

        let score_avg = score_acc/(landmarks.len() as Float);

        ((*cam_id_1, *cam_id_2), score_avg)
    }).collect()
}
