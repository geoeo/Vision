extern crate nalgebra as na;

use na::{Isometry3, Point3, Vector3,SVector, SMatrix};
use crate::sfm::landmark::Landmark;
use crate::Float;

#[derive(Copy,Clone)]
pub struct EuclideanLandmark {
    state: Point3<Float>,
}

impl Landmark<3> for EuclideanLandmark {

    fn new(state: SVector<Float,3>) -> EuclideanLandmark {
        EuclideanLandmark{state: Point3::<Float>::from(state)}
    }

    fn update(&mut self,perturb :&SVector<Float,3>) -> () {
        self.state.coords += perturb;
    }

    fn get_euclidean_representation(&self) -> Point3<Float> {
        self.state
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<Float>) -> Point3<Float> {
        other_cam_world*self.state
        
    }

    fn jacobian(&self, world_to_cam: &Isometry3<Float>) -> SMatrix<Float,3,{Self::LANDMARK_PARAM_SIZE}> {
        world_to_cam.rotation.to_rotation_matrix().matrix().into_owned()
    }

    fn get_state_as_vector(&self) -> &Vector3<Float> {
        &self.state.coords
    }

    fn from_array(arr: &[Float; Self::LANDMARK_PARAM_SIZE]) -> EuclideanLandmark {
        EuclideanLandmark{state: Point3::<Float>::new(arr[0],arr[1],arr[2])}
    }

    fn get_state_as_array(&self) -> [Float; Self::LANDMARK_PARAM_SIZE] {
        [self.state[0], self.state[1], self.state[2]]
    }
}
