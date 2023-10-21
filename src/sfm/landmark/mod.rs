extern crate nalgebra as na;

use na::{Isometry3, Point3, SVector, SMatrix,base::Scalar};

pub mod euclidean_landmark;
pub mod inverse_depth_landmark;


pub trait Landmark<F: Scalar, const T: usize> {
    const LANDMARK_PARAM_SIZE: usize = T;
    fn from_state_with_id(state: SVector<F,T>, id: &Option<usize>) -> Self;
    fn from_state(state: SVector<F, T>) -> Self; 
    fn update(&mut self, perturb :&SVector<F,T>) -> (); //TODO: change signature for inverse depth
    fn set_landmark(&mut self,l :&SVector<F,T>) -> ();
    fn get_euclidean_representation(&self) -> Point3<F>;
    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<F>) -> Point3<F>; //TODO: check this vs get_euclidean_representation
    fn jacobian(&self, world_to_cam: &Isometry3<F>) -> SMatrix<F,3,T>;
    fn get_state_as_vector(&self) -> &SVector<F, T>;
    fn from_array(arr: &[F; T]) -> Self;
    fn get_state_as_array(&self) -> [F; T];
    fn get_id(&self) -> Option<usize>;
}