extern crate nalgebra as na;

use na::{U1,U3,Vector,Vector3,Matrix2x3,Matrix3,Matrix3x4,Matrix4, base::storage::Storage};
use crate::Float;
use crate::image::features::geometry::point::Point;

pub mod pinhole;
pub mod perspective;
pub mod camera_data_frame;

pub trait Camera {
    fn get_projection(&self) -> Matrix3<Float>;
    fn get_inverse_projection(&self) -> Matrix3<Float>;
    fn get_jacobian_with_respect_to_position_in_camera_frame<T>(&self, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1>;
    fn project<T>(&self, position: &Vector<Float,U3,T>) -> Point<Float> where T: Storage<Float,U3,U1>;
    fn backproject(&self, point: &Point<Float>, depth: Float) -> Vector3<Float>;
}

/**
 * Photogrammetric Computer Vision p.498
 * Decomposes general camera projection P into K[R|t].
 * Where K is the camera intrinsics and R|t are the extrinsics.
 * 
 * Covariance propagation is not implemented.
 */
#[allow(non_snake_case)]
pub fn decompose_projection(projection_matrix: &Matrix3x4<Float>, positive_principal_distance: bool) -> (Matrix3<Float>, Matrix4<Float>) {

    let s = match positive_principal_distance {
        true => 1.0,
        false => -1.0
    };

    let A = projection_matrix.fixed_columns::<3>(0);
    let a = projection_matrix.fixed_columns::<1>(3);

    let Z = -A.try_inverse().expect("Inverse of A in decompose projection failed!")*a;
    let A_norm = match A.determinant() {
        det if det < 0.0 => -1.0*A,
        det if det > 0.0 => A.into(),
        _ => panic!("Matrix A in decompose projection is singular!")
    };
    let qr_decomp = A_norm.try_inverse().expect("Inverse of A norm in decompose projection failed!").qr();
    let mut R = qr_decomp.q().try_inverse().expect("Inverse of q in decompose projection failed");
    let mut K = qr_decomp.r().try_inverse().expect("Inverse of r in decompose projection failed");
    let K_diag = K.diagonal();
    let mut K_diag_sign = Vector3::<Float>::zeros();
    for i in 0..3 {
        K_diag_sign[i] = match K_diag[i] {
            v if v < 0.0 => -1.0,
            v if v > 0.0 => 1.0,
            _ => panic!("K diag element is 0!")
        };
    }

    let D = Matrix3::<Float>::from_diagonal(&K_diag_sign)*Matrix3::<Float>::from_diagonal(&Vector3::<Float>::new(s,s,1.0));
    R = D*R;
    K = K*D;
    K = K/K[(2,2)];

    let mut pose = Matrix4::<Float>::identity();
    pose.fixed_slice_mut::<3,3>(0,0).copy_from(&R);
    pose.fixed_slice_mut::<3,1>(0,3).copy_from(&Z);

    (K,pose)
}