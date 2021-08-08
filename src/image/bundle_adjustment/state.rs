extern crate nalgebra as na;

use na::DVector;
use crate::Float;

pub struct State {
    pub state: DVector<Float>,
    pub n_cams: usize, 
    pub n_points: usize
}