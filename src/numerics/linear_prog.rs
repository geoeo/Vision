extern crate nalgebra as na;

use na::{OMatrix, OVector, Const};
use crate::Float;

/**
 * Using Primal-Dual Algorithm. We recast L1-Norm as the sum of the inequalities i.e. Min(L1) => Min(Ax-y-e + -Ax+y-e), where Ax-y = e
 */
pub fn l1_norm_approx<const N: usize, const M: usize>(measurements: &OVector<Float,Const<N>>, design_matrix: &OMatrix<Float, Const<N>,Const<M>>,  state: & OVector<Float,Const<M>>) -> () {

    let res_primal = primal_residual(measurements,design_matrix,state);


    panic!("TODO");
}

fn primal_residual<const N: usize, const M: usize>(measurements: &OVector<Float,Const<N>>, design_matrix: &OMatrix<Float, Const<N>,Const<M>>, state: & OVector<Float,Const<M>>) -> OVector<Float,Const<N>> {
    design_matrix*state - measurements
}