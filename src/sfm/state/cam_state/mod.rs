use na::{Isometry3,SVector,SMatrix,Matrix3x6, Point3};
use crate::sensors::camera::Camera;
use crate::GenericFloat;

pub mod cam_extrinsic_state;
pub mod cam_extrinsic_intrinsic_state;

pub trait CamState<F: GenericFloat, C: Camera<F>, const CAMERA_PARAM_SIZE: usize> {
    fn new(raw_state: SVector<F,CAMERA_PARAM_SIZE>, camera: &C) -> Self;
    fn update(&mut self, perturb: SVector<F,CAMERA_PARAM_SIZE>);
    fn to_serial(&self) ->  [F; CAMERA_PARAM_SIZE];
    fn get_position(&self) -> Isometry3<F>;
    fn get_camera(&self) -> C;
    fn get_jacobian(&self, point: &Point3<F>, lie_jacobian: &Matrix3x6<F>) -> SMatrix<F,2,CAMERA_PARAM_SIZE>;
    fn duplicate(&self) -> Self;
}