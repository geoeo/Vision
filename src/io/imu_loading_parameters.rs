extern crate nalgebra as na;

use na::UnitQuaternion;
use crate::Float;
use std::fmt;

pub struct ImuLoadingParameters {
    pub invert_x: bool
}

impl fmt::Display for ImuLoadingParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invert_x_{}", self.invert_x)
    }

}