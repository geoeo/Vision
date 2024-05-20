extern crate nalgebra as na;

use na::{Matrix3,Matrix4,Matrix5,Matrix6,Vector3,Vector4,Isometry3,Matrix3x4,Matrix4x6};
use crate::Float;
use crate::image::features::{matches::Match,Feature};

/**
 * Determining an initial image pair for fixing the
 * scale of a 3d reconstruction from an image
 * sequence. Beder et al.
 */

pub fn estimate_covariance_for_match<Feat: Feature>(
        m: &Match<Feat>, 
        landmark_world: Vector4<Float>, 
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
            0.0,1.0/landmark_cam.y,0.0,-landmark_cam.y/landmark_cam.w.powi(2),
            0.0,0.0,landmark_cam.z,-landmark_cam.z/landmark_cam.w.powi(2)
        );
        j_e*cov_point*j_e.transpose()
}

/**
 * Score covariance by roundness factor. Best value for reconstruction is a score of sqrt(1/2)
 */
pub fn score_covariance(cov: &Matrix3<Float>) -> Float {
    let svd = cov.svd(false, false);
    (svd.singular_values[0] / svd.singular_values[2]).sqrt()
}
