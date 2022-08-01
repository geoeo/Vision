extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use na::{U1,U3,Vector,Vector3,Matrix2x3,Matrix3,Matrix3x4,Matrix4, base::storage::Storage, SimdRealField, ComplexField,base::Scalar};
use num_traits::{float,NumAssign};
use crate::image::features::geometry::point::Point;

pub mod pinhole;
pub mod perspective;
pub mod camera_data_frame;

 //@TODO: unify principal distance into enum
pub trait Camera<F: num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField> {
    fn get_projection(&self) -> Matrix3<F>;
    fn get_inverse_projection(&self) -> Matrix3<F>; //@TODO: rename to camera/intrinsic matrix
    fn get_jacobian_with_respect_to_position_in_camera_frame<T>(&self, position: &Vector<F,U3,T>) -> Matrix2x3<F> where T: Storage<F,U3,U1>;
    fn project<T>(&self, position: &Vector<F,U3,T>) -> Point<F> where T: Storage<F,U3,U1>;
    fn backproject(&self, point: &Point<F>, depth: F) -> Vector3<F>;
}

/**
 * Photogrammetric Computer Vision p.498
 * Decomposes general camera projection P into K[R|t].
 * Where K is the camera intrinsics and R|t are the extrinsics.
 * 
 * Covariance propagation is not implemented.
 * @TODO: unify principal distance into enum
 */
#[allow(non_snake_case)]
pub fn decompose_projection<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField>(projection_matrix: &Matrix3x4<F>, positive_principal_distance: bool) -> (Matrix3<F>, Matrix4<F>) {

    let s = match positive_principal_distance {
        true => F::one(),
        false => -F::one()
    };

    let A = projection_matrix.fixed_columns::<3>(0);
    let a = projection_matrix.fixed_columns::<1>(3);

    let Z = -A.try_inverse().expect("Inverse of A in decompose projection failed!")*a;
    let A_norm = match A.determinant() {
        det if det < F::zero() => A.scale(-F::one()),
        det if det > F::zero() => A.into(),
        _ => panic!("Matrix A in decompose projection is singular!")
    };
    let qr_decomp = A_norm.try_inverse().expect("Inverse of A norm in decompose projection failed!").qr();
    let mut R = qr_decomp.q().try_inverse().expect("Inverse of q in decompose projection failed");
    let mut K = qr_decomp.r().try_inverse().expect("Inverse of r in decompose projection failed");
    let K_diag = K.diagonal();
    let mut K_diag_sign = Vector3::<F>::zeros();
    for i in 0..3 {
        K_diag_sign[i] = match K_diag[i] {
            v if v < F::zero() => -F::one(),
            v if v > F::zero() => F::one(),
            _ => panic!("K diag element is 0!")
        };
    }

    let D = Matrix3::<F>::from_diagonal(&K_diag_sign)*Matrix3::<F>::from_diagonal(&Vector3::<F>::new(s,s,F::one()));
    R = D*R;
    K = K*D;
    K = K/K[(2,2)];

    let mut pose = Matrix4::<F>::identity();
    pose.fixed_slice_mut::<3,3>(0,0).copy_from(&R);
    pose.fixed_slice_mut::<3,1>(0,3).copy_from(&Z);

    (K,pose)
}