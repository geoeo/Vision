extern crate nalgebra as na;

use na::{Matrix3, Isometry3, Point3};
use crate::Float;

#[derive(Copy,Clone)]
pub struct EuclideanLandmark {
    state: Point3<Float>,
}

impl EuclideanLandmark {

    pub const LANDMARK_PARAM_SIZE: usize = 3;

    pub fn new(x: Float, y: Float, z: Float) -> EuclideanLandmark {
        EuclideanLandmark{state: Point3::<Float>::new(x,y,z)}
    }

    pub fn update(&mut self,delta_x: Float, delta_y: Float, delta_z: Float) -> () {
        self.state.x += delta_x;
        self.state.y += delta_y;
        self.state.z += delta_z;
    }

    pub fn get_euclidean_representation(&self) -> Point3<Float> {
        self.state
    }

    pub fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<Float>) -> Point3<Float> {
        other_cam_world*self.state
        
    }

    pub fn jacobian(&self, world_to_cam: &Isometry3<Float>) -> Matrix3<Float> {
        world_to_cam.rotation.to_rotation_matrix().matrix().into_owned()
    }

    pub fn get_state(&self) -> &Point3<Float> {
        &self.state
    }
}