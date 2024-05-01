use na::{Isometry3,SVector, base::Scalar};

pub mod cam_extrinsic_state;

pub trait CamState<F: Scalar, const CAMERA_PARAM_SIZE: usize> {
    const CAMERA_PARAM_SIZE: usize;
    fn new(raw_state: SVector<F,CAMERA_PARAM_SIZE>) -> Self;
    fn update(&mut self, perturb: SVector<F,CAMERA_PARAM_SIZE>);
    fn to_serial(&self) ->  [F; CAMERA_PARAM_SIZE];
    fn get_position(&self) -> Isometry3<F>;
    fn duplicate(&self) -> Self;
}