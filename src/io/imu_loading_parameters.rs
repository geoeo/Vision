extern crate nalgebra as na;

use na::UnitQuaternion;
use crate::Float;
use std::fmt;

pub struct ImuLoadingParameters {
    pub accel_invert_x: bool,
    pub sensor_alignment_rot: UnitQuaternion<Float>
}

impl fmt::Display for ImuLoadingParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invert_x_{}", self.accel_invert_x)
    }

}