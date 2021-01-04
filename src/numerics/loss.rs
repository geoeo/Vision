extern crate nalgebra as na;

use na::DVector;
use crate::Float;


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