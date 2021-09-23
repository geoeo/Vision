extern crate nalgebra as na;

use na::DVector;
use crate::Float;

/**
 * This is ordered [cam_1,cam_2,..,cam_n,point_1,point_2,...,point_m]
 * cam is parameterized by [u_1,u_2,u_3,w_1,w_2,w_3]
 * point is parameterized by [x,y,z]
 * */
#[derive(Clone)]
pub struct State {
    pub data: DVector<Float>,
    pub n_cams: usize, 
    pub n_points: usize
}

impl State {
    pub fn update(&mut self, perturb: &DVector<Float>) -> (){
       self.data += perturb;
    }
}