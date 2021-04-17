extern crate nalgebra as na;

use na::UnitQuaternion;
use crate::Float;
use std::fmt;

pub struct ImuLoadingParameters {
    pub convert_to_cam_coords: bool,
    pub sensor_alignment_rot: UnitQuaternion<Float>
}

impl fmt::Display for ImuLoadingParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "convert_to_cam_coords_{}", self.convert_to_cam_coords)
    }

}