extern crate nalgebra as na;
extern crate num_traits;

use na::{Isometry3, Point3, Vector3,SVector, SMatrix, RealField, base::Scalar};
use num_traits::{float,NumAssign};
use std::marker::Send;
use crate::sfm::landmark::Landmark;

#[derive(Copy,Clone)]
pub struct EuclideanLandmark<F: float::Float + Scalar + NumAssign + RealField + Send> {
    state: Point3<F>,
}

impl<F: float::Float + Scalar + NumAssign + RealField> Landmark<F,3> for EuclideanLandmark<F> {

    fn from_state(state: SVector<F,3>) -> EuclideanLandmark<F> {
        EuclideanLandmark{state: Point3::<F>::from(state)}
    }

    fn update(&mut self,perturb :&SVector<F,3>) -> () {
        self.state.coords += perturb;
    }

    fn get_euclidean_representation(&self) -> Point3<F> {
        self.state
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<F>) -> Point3<F> {
        other_cam_world*self.state
        
    }

    fn jacobian(&self, world_to_cam: &Isometry3<F>) -> SMatrix<F,3,3> {
        world_to_cam.rotation.to_rotation_matrix().matrix().into_owned()
    }

    fn get_state_as_vector(&self) -> &Vector3<F> {
        &self.state.coords
    }

    fn from_array(arr: &[F; 3]) -> EuclideanLandmark<F> {
        EuclideanLandmark{state: Point3::<F>::new(arr[0],arr[1],arr[2])}
    }

    fn get_state_as_array(&self) -> [F; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }
}
