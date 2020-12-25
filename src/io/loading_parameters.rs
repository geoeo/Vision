extern crate nalgebra as na;

use na::UnitQuaternion;
use crate::Float;

pub struct LoadingParameters {
    pub starting_index: usize,
    pub step: usize,
    pub count: usize,
    pub negate_values: bool,
    pub invert_focal_lengths: bool,
    pub invert_y: bool,
    pub gt_alignment: UnitQuaternion<Float>
}