extern crate nalgebra as na;

use na::{OMatrix, OVector, DVector, Const, Vector2};
use crate::Float;

/**
 * Using Primal-Dual Algorithm. We recast L1-Norm as the sum of the inequalities i.e. Min(L1) => Min(Ax-y-f + -Ax+y-f), where Ax-y = e
 */

#[allow(non_snake_case)]
pub fn l1_norm_approx<const N: usize, const M: usize>(measurements: &OVector<Float,Const<N>>, A: &OMatrix<Float, Const<N>,Const<M>>,  state: &OVector<Float,Const<M>>) -> () {

    let alpha = 0.01;
    let beta = 0.5;
    let mu = 10.0;
    let m = 2.0;

    let res_primal = primal_residual(measurements,A,state);
    let h_init = compute_h(measurements,A,state,&res_primal);
    let mut u_vec = h_init.map(|v| v.recip());

    // We only have dual state for inequalities (no nu vector)
    let mut dual_state = Vector2::<Float>::zeros();




    panic!("TODO");
}

#[allow(non_snake_case)]
fn primal_residual<const N: usize, const M: usize>(measurements: &OVector<Float,Const<N>>, A: &OMatrix<Float, Const<N>,Const<M>>, state: & OVector<Float,Const<M>>) -> OVector<Float,Const<N>> {
    A*state - measurements
}

#[allow(non_snake_case)]
fn compute_f<const N: usize, const M: usize>(measurements: &OVector<Float,Const<N>>, A: &OMatrix<Float, Const<N>,Const<M>>,  state: &OVector<Float,Const<M>>) -> OVector<Float,Const<N>> {
    let res_abs = (A*state-measurements).abs();
    (0.95*res_abs).add_scalar(0.1*res_abs.max())
}

#[allow(non_snake_case)]
fn compute_h<const N: usize, const M: usize>(measurements: &OVector<Float,Const<N>>, A: &OMatrix<Float, Const<N>,Const<M>>, state: &OVector<Float,Const<M>>, residual: &OVector<Float,Const<N>>) -> DVector<Float> {
    let f_1 = A*state-measurements - residual;
    let f_2 = -A*state+measurements -residual;
    let mut h = DVector::<Float>::zeros(N*2);
    h.fixed_rows_mut::<N>(0).copy_from(&f_1);
    h.fixed_rows_mut::<N>(N).copy_from(&f_2);
    h
}

fn compute_eta<const H: usize>(h: &OVector<Float,Const<H>>, us: &OVector<Float,Const<H>>) -> Float {
    (-h.transpose()*us)[(0,0)]
}

