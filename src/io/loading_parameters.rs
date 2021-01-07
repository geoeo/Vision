extern crate nalgebra as na;

use na::UnitQuaternion;
use crate::Float;
use std::fmt;

pub struct LoadingParameters {
    pub starting_index: usize,
    pub step: usize,
    pub count: usize,
    pub negate_depth_values: bool,
    pub invert_focal_lengths: bool,
    pub invert_y: bool,
    pub gt_alignment_rot: UnitQuaternion<Float>
}

impl fmt::Display for LoadingParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "starting_index_{}_step_{}_count_{}_negate_values_{}_invert_focal_lengths_{}_invert_y_{}", 
        self.starting_index,self.step,self.count,self.negate_depth_values,self.invert_focal_lengths,self.invert_y)
    }

}