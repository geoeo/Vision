extern crate nalgebra as na;

use na::{OMatrix, OVector, DVector, DMatrix, Const, Vector2};
use crate::Float;

/**
 * Using Primal-Dual Algorithm. We recast L1-Norm as the sum of the inequalities i.e. Min(L1) => Min x,c (c (= f0)) subject to Ax-y-c <= 0 and + -Ax+y-c <=0 
 */

#[allow(non_snake_case)]
pub fn l1_norm_approx<const N: usize, const M: usize>(measurements: &OVector<Float,Const<M>>, A: &OMatrix<Float, Const<M>,Const<N>>,  state: &mut OVector<Float,Const<N>>) -> () {

    let max_iter = 200;
    let tol = 1e-3;

    let alpha = 0.01;
    let beta = 0.5;
    let mu = 10.0;
    let dual_dim = (2*M) as Float;

    // Since f_0 = u => du/dx = 0, du/du = 1
    let f_0_grad_x = DVector::<Float>::zeros(N);
    let f_0_grad_u = DVector::<Float>::zeros(M);
    let mut f_0_grad = DVector::<Float>::zeros(N+M);
    f_0_grad.fixed_rows_mut::<N>(0).copy_from(&f_0_grad_x);
    f_0_grad.fixed_rows_mut::<M>(N).copy_from(&f_0_grad_u);
    let A_t = A.transpose();
    let ones = DVector::<Float>::repeat(N*2,1.0);



    let mut c = f_0(measurements,A,state);
    let mut h = h(measurements,A,state,&c);
    let mut h_recip = h.map(|v| v.recip());
    let mut u_vec = u(&h,1.0); //lamu in l1magic 

    let eta = eta(&h,&u_vec); //sdg (surrogate duality gap)
    let tau = tau(mu,dual_dim as Float,eta);


    let mut iter = 0;


    while iter < max_iter && eta < tol {

        let w2 = w2(&u_vec,&ones);
        let (sigma_1,sigma_2,sigma_x) = sigmas(&u_vec, &h_recip);


    }


    panic!("TODO");
}

//Since we dont have an equality constraint we do not have a nu term
#[allow(non_snake_case)]
fn dual_residual<const N: usize, const M: usize>(f_0_grad: &DVector<Float>,  A_t: &OMatrix<Float, Const<N>,Const<M>>, u_vec: &DVector<Float>) -> DVector<Float> {
    let mut dh_u = DVector::<Float>::zeros(N+M);
    let lamu_1 = u_vec.fixed_rows::<M>(0);
    let lamu_2 = u_vec.fixed_rows::<M>(M);

    dh_u.fixed_rows_mut::<N>(0).copy_from(&(A_t*(lamu_1-lamu_2)));
    dh_u.fixed_rows_mut::<M>(N).copy_from(&(-lamu_1-lamu_2));

    f_0_grad+dh_u
}

#[allow(non_snake_case)]
fn center_residual<>(h: &DVector<Float>, u_vec: &DVector<Float>, tau: Float) -> DVector<Float> {
    let mut diag = DMatrix::<Float>::zeros(u_vec.nrows(),u_vec.nrows());
    diag.set_diagonal(u_vec);
    let mut center_residual = -diag*h;
    center_residual.add_scalar_mut(-1.0/tau);
    center_residual
}

#[allow(non_snake_case)]
fn f_0<const N: usize, const M: usize>(measurements: &OVector<Float,Const<M>>, A: &OMatrix<Float, Const<M>,Const<N>>,  state: &OVector<Float,Const<N>>) -> OVector<Float,Const<M>> {
    let res_abs = (A*state-measurements).abs();
    (0.95*res_abs).add_scalar(0.1*res_abs.max())
}

#[allow(non_snake_case)]
fn h<const N: usize, const M: usize>(measurements: &OVector<Float,Const<M>>, A: &OMatrix<Float, Const<M>,Const<N>>, state: &OVector<Float,Const<N>>, residual: &OVector<Float,Const<M>>) -> DVector<Float> {
    let f_1 = A*state-measurements - residual;
    let f_2 = -A*state+measurements -residual;
    let mut h = DVector::<Float>::zeros(M*2);
    h.fixed_rows_mut::<M>(0).copy_from(&f_1);
    h.fixed_rows_mut::<M>(M).copy_from(&f_2);
    h
}

fn eta(h: &DVector<Float>, u_vec: &DVector<Float>) -> Float {
    (-h.transpose()*u_vec)[(0,0)]
}

fn tau(mu: Float, m: Float, eta: Float) -> Float {
    mu*m/eta
}

fn u(h: &DVector<Float>, tau: Float) -> DVector<Float> {
    h.map(|v| -(v*tau).recip())
}

fn w2(u: &DVector<Float>,ones: &DVector<Float>) -> DVector<Float> {
    ones-u
}

/**
 * (sigma_1,sigma_2,sigma_x)
 */
fn sigmas(u: &DVector<Float>, h_recip: &DVector<Float>) -> (DVector<Float>,DVector<Float>,DVector<Float>) {
    let subproblem_size = h_recip.nrows()/2;
    let v = u*h_recip;
    let v_1 = v.rows(0,subproblem_size);
    let v_2 = v.rows(subproblem_size,subproblem_size);

    let sigma_1 = -v_1 - v_2;
    let sigma_1_recip = sigma_1.map(|x| x.recip());
    let sigma_2 = v_1 - v_2;
    let sigma_2_squared = sigma_2.map(|x| x.powi(2));

    let sigma_x = sigma_1 - sigma_2_squared.component_mul(&sigma_1_recip);
    (sigma_1,sigma_2,sigma_x)

}

