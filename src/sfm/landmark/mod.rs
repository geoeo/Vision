extern crate nalgebra as na;
extern crate num_traits;

use na::{Isometry3, Point3, SVector, SMatrix, SimdRealField, ComplexField,base::Scalar};
use num_traits::{float,NumAssign};

pub mod euclidean_landmark;
pub mod inverse_depth_landmark;


pub trait Landmark<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField, const T: usize> {
    const LANDMARK_PARAM_SIZE: usize = T;

    fn from_state(state: SVector<F, T>) -> Self; 
    fn update(&mut self, perturb :&SVector<F,T>) -> (); //TODO: change signature for inverse depth
    fn get_euclidean_representation(&self) -> Point3<F>;
    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<F>) -> Point3<F>; //TODO: check this vs get_euclidean_representation
    fn jacobian(&self, world_to_cam: &Isometry3<F>) -> SMatrix<F,3,T>;
    fn get_state_as_vector(&self) -> &SVector<F, T>;
    fn from_array(arr: &[F; T]) -> Self;
    fn get_state_as_array(&self) -> [F; T];
}