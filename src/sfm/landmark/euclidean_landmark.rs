extern crate nalgebra as na;

use na::{Isometry3, Point3, Vector3,SVector, SMatrix, RealField, base::Scalar};
use crate::sfm::landmark::Landmark;

#[derive(Copy,Clone)]
pub struct EuclideanLandmark<F:  Scalar + RealField + Copy> {
    state: Point3<F>,
    id: Option<usize>
}

impl<F: Scalar  + RealField + Copy> Landmark<F,3> for EuclideanLandmark<F> {

    fn from_state_with_id(state: SVector<F,3>, id: &Option<usize>) -> EuclideanLandmark<F> {
        EuclideanLandmark{state: Point3::<F>::from(state), id: *id}
    }

    fn from_state(state: SVector<F,3>) -> EuclideanLandmark<F> {
        EuclideanLandmark{state: Point3::<F>::from(state), id: None}
    }

    fn update(&mut self,perturb :&SVector<F,3>) -> () {
        self.state.coords += perturb;
    }

    fn set_state(&mut self, state :&SVector<F,3>) -> () {
        self.state = Point3::<F>::new(state.x,state.y,state.z);
    }

    fn get_euclidean_representation(&self) -> Point3<F> {
        self.state
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<F>) -> EuclideanLandmark<F> {
        let new_pos = other_cam_world*self.state;
        EuclideanLandmark::<F>{state: new_pos, id: self.id}
    }

    fn jacobian(&self, world_to_cam: &Isometry3<F>) -> SMatrix<F,3,3> {
        world_to_cam.rotation.to_rotation_matrix().matrix().into_owned()
    }

    fn get_state_as_vector(&self) -> &Vector3<F> {
        &self.state.coords
    }

    fn from_array(arr: &[F; 3]) -> EuclideanLandmark<F> {
        EuclideanLandmark{state: Point3::<F>::new(arr[0],arr[1],arr[2]), id: None}
    }

    fn get_state_as_array(&self) -> [F; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    fn get_id(&self) -> Option<usize> {self.id}
}
