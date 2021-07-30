extern crate nalgebra as na;

use std::fmt;

pub struct ImuLoadingParameters {
    pub convert_to_cam_coords: bool
}

impl fmt::Display for ImuLoadingParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "convert_to_cam_coords_{}", self.convert_to_cam_coords)
    }

}