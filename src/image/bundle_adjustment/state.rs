extern crate nalgebra as na;

use na::{DVector,Matrix4, Vector3};
use crate::Float;
use crate::numerics::lie::{exp};

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


    pub fn lift(&self) -> (Vec<Matrix4<Float>>,Vec<Vector3<Float>>) {

        let mut cam_positions = Vec::<Matrix4<Float>>::with_capacity(self.n_cams);
        let mut points = Vec::<Vector3<Float>>::with_capacity(self.n_points);

        for i in (0..6*self.n_cams).step_by(6) {
            let u = self.data.fixed_rows::<3>(i);
            let w = self.data.fixed_rows::<3>(i+3);
            cam_positions.push(exp(&u,&w));
        }

        
        for i in (6*self.n_cams..self.data.nrows()).step_by(3) {
            let point = self.data.fixed_rows::<3>(i);
            points.push(Vector3::from(point));
        }


        (cam_positions,points)
    }
}