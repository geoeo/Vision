extern crate nalgebra as na;

use na::DVector;
use crate::Float;


pub trait LossFunction {
    fn is_valid(&self) -> bool {
        2.0*self.second_derivative_at_current()*self.current_cost()+self.first_derivative_at_current() > 0.0 + self.eps()
    }
    fn eps(&self) -> Float;
    fn root_approx(&self) -> (Float,Float) {
        let v = 1.0 - self.eps();
        (v,v)
    }
    fn update(&self,current_residual: Float) -> ();
    fn first_derivative_at_current(&self) -> Float;
    fn second_derivative_at_current(&self) -> Float;
    fn current_cost(&self) -> Float;
}


//TODO: Loss impls
//Not correct weight != loss! http://ceres-solver.org/nnls_modeling.html#theory
fn compute_cauchy_loss(residuals: &DVector<Float>, weights_vec: &mut DVector<Float>) -> () {
    for i in 0..residuals.len() {
        weights_vec[i] = (1.0+residuals[i]).ln();
    }
}

fn compute_huber_loss(residuals: &DVector<Float>, weights_vec: &mut DVector<Float>) -> () {
    for i in 0..residuals.len() {
        let res = residuals[i];
        weights_vec[i] = match res {
            res if res <= 1.0 => res,
            _ => 2.0*res.sqrt()-1.0
        }

    }
}