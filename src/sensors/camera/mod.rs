extern crate nalgebra as na;
extern crate simba;

use std::collections::HashMap;
use na::{U1,U3,Vector,Vector3,Matrix2x3,Matrix2x5,Matrix3,Matrix3x4,Matrix4, base::storage::Storage};
use simba::scalar::SupersetOf;
use crate::image::features::geometry::point::Point;
use crate::GenericFloat;

pub mod perspective;

#[derive(Hash,PartialEq,Eq,Copy,Clone)]
pub enum INTRINSICS {
    FX,
    FY,
    CX,
    CY,
    S
}

 //@TODO: unify principal distance into enum
 pub trait Camera<F: GenericFloat> {
    fn get_projection(&self) -> Matrix3<F>;
    fn get_inverse_projection(&self) -> Matrix3<F>; //@TODO: rename to camera/intrinsic matrix
    fn get_jacobian_with_respect_to_position_in_camera_frame<T, F2: GenericFloat + SupersetOf<F>>(&self, position: &Vector<F2,U3,T>) -> Option<Matrix2x3<F2>> where T: Storage<F2,U3,U1>;
    fn get_jacobian_with_respect_to_intrinsics<T, F2: GenericFloat + SupersetOf<F>>(&self, position: &Vector<F2,U3,T>) -> Option<Matrix2x5<F2>> where T: Storage<F2,U3,U1>;
    fn project<T, F2: GenericFloat + SupersetOf<F>>(&self, position: &Vector<F2,U3,T>) -> Option<Point<F2>> where T: Storage<F2,U3,U1>;
    fn backproject(&self, point: &Point<F>, depth: F) -> Vector3<F>;
    fn get_focal_x(&self) -> F;
    fn get_focal_y(&self) -> F;
    fn from_matrices(projection: &Matrix3<F>, inverse_projection: &Matrix3<F>) -> Self;
    fn update(&mut self,perturb: &HashMap<INTRINSICS,F>) -> ();
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
pub fn decompose_projection<F: GenericFloat>(projection_matrix: &Matrix3x4<F>, positive_principal_distance: bool) -> (Matrix3<F>, Matrix4<F>) {

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
    pose.fixed_view_mut::<3,3>(0,0).copy_from(&R);
    pose.fixed_view_mut::<3,1>(0,3).copy_from(&Z);

    (K,pose)
}