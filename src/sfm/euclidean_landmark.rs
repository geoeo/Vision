extern crate nalgebra as na;

use na::{Vector3,Matrix3, Isometry3, Point3};
use crate::Float;

pub struct EuclideanLandmark {
    state: Point3<Float>,
}

impl EuclideanLandmark {
    fn new(negative_depth: bool) -> EuclideanLandmark {
        let depth = match negative_depth {
            true => -0.1,
            false => 0.1
        };
        EuclideanLandmark{state: Point3::<Float>::new(0.0,0.0,depth)}
    }

    fn get_euclidean_representation(&self) -> Point3<Float> {
        self.state
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<Float>) -> Point3<Float> {
        other_cam_world*self.state
        
    }

    fn jacobian(&self, world_to_cam: &Isometry3<Float>) -> Matrix3<Float> {
        world_to_cam.rotation.to_rotation_matrix().matrix().into_owned()
    }
}