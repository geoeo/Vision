extern crate nalgebra as na;

use na::{DVector, DMatrix};
use crate::Float;

/**
 * Using Primal-Dual Algorithm. We recast L1-Norm as the sum of the inequalities i.e. Min(L1) => Min x,c (c (= f0)) subject to Ax-y-c <= 0 and + -Ax+y-c <=0 
 */

#[allow(non_snake_case)]
pub fn l1_norm_approx(measurements: &DVector<Float>, A: &DMatrix<Float>,  x: &mut DVector<Float>, max_iter: usize, tol: Float) -> () {
    println!("WARNING: DOESNT COVERGE; STILL BUGGY");
    let M = measurements.nrows();
    let N = x.nrows();

    let alpha = 0.01;
    let beta = 0.5;
    let mu = 10.0;
    let dual_dim = (2*M) as Float;

    // Since f_0 = u => du/dx = 0, du/du = 1
    let f_0_grad_x = DVector::<Float>::zeros(N);
    let f_0_grad_u = DVector::<Float>::zeros(M);
    let mut f_0_grad = DVector::<Float>::zeros(N+M);
    f_0_grad.rows_mut(0,N).copy_from(&f_0_grad_x);
    f_0_grad.rows_mut(N,M).copy_from(&f_0_grad_u);
    let A_t = A.transpose();
    let ones = DVector::<Float>::repeat(M,1.0);
    let mut dual_residuals = DVector::<Float>::zeros(N+M+2*M);

    let mut c = f_0(measurements,A,x);
    let mut h = h(measurements,A,x,&c);
    let mut h_recip = h.map(|v| v.recip());
    //TODO: refactor this into returning u1,u2 explicitly
    let mut u_vec = u(&h_recip,1.0); //lamu in l1magic 

    let mut eta = compute_eta(&h,&u_vec); //sdg (surrogate duality gap)
    let mut tau = compute_tau(mu,dual_dim as Float,eta);
    let mut res_dual = dual_residual(&f_0_grad,&A_t,&u_vec);
    let mut res_center = center_residual(&h,&u_vec, tau);
    dual_residuals.rows_mut(0,N+M).copy_from(&res_dual);
    dual_residuals.rows_mut(N+M,2*M).copy_from(&res_center);

    let mut iter = 0;

    //TODO: doesnt converge
    while iter < max_iter && eta >= tol {
        let w1 = w1(&h_recip,&A_t, tau);
        let w2 = w2(&u_vec,&ones, tau);
        let (sigma_1_recip,sigma_2,sigma_x) = sigmas(&u_vec, &h_recip);

        let w1p = w1 - (&A_t)*(sigma_2.component_mul(&sigma_1_recip)).component_mul(&w2);
        let H11p = &A_t*DMatrix::<Float>::from_diagonal(&sigma_x)*A;
        let dx = H11p.cholesky().expect("H11p cholesky failed").solve(&w1p);
        let Adx = A*(&dx);
        let dc = w2 - (sigma_2.component_mul(&Adx)).component_mul(&sigma_1_recip);
        let (du1,du2) = du(&u_vec, &h_recip, &Adx, &dc, tau);
        let Ax = A*(x as &DVector<Float>);
        let s = s(&u_vec,&du1,&du2,&Adx,&dc,&h);
        let u1 = u_vec.rows(0,M).into_owned();
        let u2 = u_vec.rows(M,M).into_owned();
        let (xp,cp,u1p,u2p,h1p,h2p,rdp) = backtrack_line_search(x,&dx,&c,&dc,&Ax,&Adx,&A_t,&u1,&u2,&du1,&du2,measurements,&f_0_grad, dual_residuals.norm(),alpha,beta,s, tau);

        x.copy_from(&xp);
        c.copy_from(&cp);
        u_vec.rows_mut(0,M).copy_from(&u1p);
        u_vec.rows_mut(M,M).copy_from(&u2p);
        h.rows_mut(0,M).copy_from(&h1p);
        h.rows_mut(M,M).copy_from(&h2p);
        h_recip = h.map(|v| v.recip());

        eta = compute_eta(&h,&u_vec); 
        tau = compute_tau(mu,dual_dim as Float,eta);
        res_dual = rdp;
        res_center = center_residual(&h,&u_vec, tau);
        dual_residuals.rows_mut(0,N+M).copy_from(&res_dual);
        dual_residuals.rows_mut(N+M,2*M).copy_from(&res_center);

        println!("eta: {}", eta);

        iter+=1;

    }

    //TODO: wrap in debug mode
    println!("Iterations: {}, tau = {} Primal = {}, PDGap = {}, Dual res = {}", iter,tau, u_vec.sum(), eta, res_dual.norm());
}

//Since we dont have an equality constraint we do not have a nu term
#[allow(non_snake_case)]
fn dual_residual(f_0_grad: &DVector<Float>,  A_t: &DMatrix<Float>, u_vec: &DVector<Float>) -> DVector<Float> {

    let M = A_t.ncols();
    let N = A_t.nrows();


    let mut dh_u = DVector::<Float>::zeros(N+M);
    let lamu_1 = u_vec.rows(0,M);
    let lamu_2 = u_vec.rows(M,M);

    dh_u.rows_mut(0,N).copy_from(&(A_t*(lamu_1-lamu_2)));
    dh_u.rows_mut(N,M).copy_from(&(-lamu_1-lamu_2));

    f_0_grad+dh_u
}

#[allow(non_snake_case)]
fn center_residual(h: &DVector<Float>, u_vec: &DVector<Float>, tau: Float) -> DVector<Float> {
    let mut center_residual = (-u_vec).component_mul(&h);
    center_residual.add_scalar_mut(-1.0/tau);
    center_residual
}

#[allow(non_snake_case)]
fn f_0(measurements: &DVector<Float>, A: &DMatrix<Float>,  state: &DVector<Float>) -> DVector<Float> {
    let res_abs = (A*state-measurements).abs();
    (0.95*(&res_abs)).add_scalar(0.1*(&res_abs).max())
}

#[allow(non_snake_case)]
fn h(measurements: &DVector<Float>, A: &DMatrix<Float>, state: &DVector<Float>, residual: &DVector<Float>) -> DVector<Float> {
    let M = measurements.nrows();

    let f_1 = A*state-measurements - residual;
    let f_2 = -A*state+measurements -residual;
    let mut h = DVector::<Float>::zeros(M*2);
    h.rows_mut(0,M).copy_from(&f_1);
    h.rows_mut(M,M).copy_from(&f_2);
    h
}

fn compute_eta(h: &DVector<Float>, u_vec: &DVector<Float>) -> Float {
    (-h.transpose()*u_vec)[(0,0)]
}

fn compute_tau(mu: Float, m: Float, eta: Float) -> Float {
    mu*m/eta
}

fn u(h_recip: &DVector<Float>, tau: Float) -> DVector<Float> {
    h_recip*-1.0/tau
}

#[allow(non_snake_case)]
fn w1(h_recip: &DVector<Float>, A_t: &DMatrix<Float>, tau: Float) -> DVector<Float> {
    let subproblem_size = h_recip.nrows()/2;
    let h_1 = h_recip.rows(0,subproblem_size);
    let h_2 = h_recip.rows(subproblem_size,subproblem_size);

    (-1.0/tau)*A_t*(-h_1+h_2)


}

fn w2(h_recip: &DVector<Float>,ones: &DVector<Float>, tau: Float) -> DVector<Float> {
    let subproblem_size = h_recip.nrows()/2;
    let h_1 = h_recip.rows(0,subproblem_size);
    let h_2 = h_recip.rows(subproblem_size,subproblem_size);
    -ones-(h_1+h_2)*1.0/tau
}

/**
 * (sigma_1_recip,sigma_2,sigma_x)
 */
fn sigmas(u: &DVector<Float>, h_recip: &DVector<Float>) -> (DVector<Float>,DVector<Float>,DVector<Float>) {
    let subproblem_size = h_recip.nrows()/2;
    let uh = u.component_mul(&h_recip);
    let uh_1 = uh.rows(0,subproblem_size);
    let uh_2 = uh.rows(subproblem_size,subproblem_size);

    let sigma_1 = -uh_1 - uh_2;
    let sigma_1_recip = sigma_1.map(|x| x.recip());
    let sigma_2 = uh_1 - uh_2;
    let sigma_2_squared = sigma_2.map(|x| x.powi(2));

    let sigma_x = &sigma_1 - sigma_2_squared.component_mul(&sigma_1_recip);
    (sigma_1_recip,sigma_2,sigma_x)
}
/**
 * (du1,du2)
 */
#[allow(non_snake_case)]
fn du(u_vec: &DVector<Float>, h_recip: &DVector<Float>, Adx: &DVector<Float>, dc: &DVector<Float>, tau: Float) -> (DVector<Float>,DVector<Float>) {
    let subproblem_size = h_recip.nrows()/2;
    let h_recip_1 = h_recip.rows(0,subproblem_size);
    let h_recip_2 = h_recip.rows(subproblem_size,subproblem_size);
    let u_1 = u_vec.rows(0,subproblem_size);
    let u_2 = u_vec.rows(subproblem_size,subproblem_size);

    (
        -(u_1.component_mul(&h_recip_1)).component_mul(&(Adx-dc))-u_1 -(1.0/tau)*h_recip_1,
        (u_2.component_mul(&h_recip_2)).component_mul(&(Adx+dc))-u_2 -(1.0/tau)*h_recip_2
    )

}

#[allow(non_snake_case)]
fn s(u_vec: &DVector<Float>, du1: &DVector<Float>, du2: &DVector<Float>, Adx: &DVector<Float>, dc: &DVector<Float>, h: &DVector<Float>) -> Float {
    let M = dc.nrows();
    let u1 = u_vec.rows(0,M);
    let u2 = u_vec.rows(M,M);
    let h1 = h.rows(0,M);
    let h2 = h.rows(M,M);

    let min_u1 = ((-u1).component_mul(&du1)).min();
    let min_u2 = ((-u2).component_mul(&du2)).min();
    let min_s1 = vec!(1.0,min_u1,min_u2).iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("min s1 failed").clone();
    let min_h1 = (-h1).component_mul(&(Adx-dc)).min();
    let min_h2 = (-h2).component_mul(&(-Adx-dc)).min();
    0.99*vec![min_s1,min_h1,min_h2].iter().min_by(|a,b|a.partial_cmp(b).unwrap()).expect("min s failed").clone()

}

#[allow(non_snake_case)]
fn backtrack_line_search(
    x: &DVector<Float>, 
    dx: &DVector<Float>,
    c:  &DVector<Float>,
    dc: &DVector<Float>,
    Ax: &DVector<Float>,
    Adx: &DVector<Float>,
    A_t: &DMatrix<Float>,
    u1: &DVector<Float>,
    u2: &DVector<Float>,
    du1: &DVector<Float>,
    du2: &DVector<Float>,
    measurements: &DVector<Float>,
    f_0_grad: &DVector<Float>,
    res_norm: Float,
    alpha: Float, beta : Float, s_init: Float,tau: Float) -> (
        DVector<Float>,
        DVector<Float>,
        DVector<Float>,
        DVector<Float>,
        DVector<Float>,
        DVector<Float>,
        DVector<Float>) {
    let M = dc.nrows();
    let N = x.nrows();

    let mut s = s_init;
    let mut suff_dec = false;
    let mut backiter = 0;
    let Atv = A_t*(u1-u2);
    let Atdv = A_t*(du1-du2);

    let mut rdp_temp = DVector::<Float>::zeros(N+M);
    let mut rcp = DVector::<Float>::zeros(2*M);
    let mut resp = DVector::<Float>::zeros(N+M+2*M);

    let mut xp = DVector::zeros(N);
    let mut cp = DVector::zeros(M);
    let mut Axp = DVector::zeros(M);
    let mut Atvp =  DVector::zeros(N);
    let mut u1p =  DVector::zeros(M);
    let mut u2p =  DVector::zeros(M);
    let mut h1p =  DVector::zeros(M);
    let mut h2p =  DVector::zeros(M);
    
    //println!("backtracking start");
    while !suff_dec && backiter <= 32 {
        xp.copy_from(&(x+s*dx));
        cp.copy_from(&(c+s*dc));
        Axp.copy_from(&(Ax+s*Adx));
        Atvp.copy_from(&((&Atv)+s*(&Atdv)));
        u1p.copy_from(&(u1+s*du1));
        u2p.copy_from(&(u2+s*du2));
        h1p.copy_from(&(&Axp - measurements - &cp));
        h2p.copy_from(&(-(&Axp) + measurements - &cp));

        rdp_temp.rows_mut(0,N).copy_from(&Atvp);
        rdp_temp.rows_mut(N,M).copy_from(&(-(&u1p)-(&u2p)));
        resp.rows_mut(0,N+M).copy_from(&(f_0_grad + &rdp_temp));

        rcp.rows_mut(0,M).copy_from(&(-(&u1p).component_mul(&h1p)));
        rcp.rows_mut(M,M).copy_from(&(-(&u2p).component_mul(&h2p)));
        rcp.add_scalar_mut(-1.0/tau);
        resp.rows_mut(N+M,2*M).copy_from(&rcp);
        let resp_norm = resp.norm();
        //println!("resp: {}",resp_norm);
        suff_dec = resp_norm <= (1.0-alpha*s)*res_norm;
        s *= beta;
        backiter+=1;
        if backiter == 32 {
            xp.copy_from(x);
        }
    }
    //println!("backtracking end");

    (xp,cp,u1p,u2p,h1p,h2p,resp.rows(0,N+M).into_owned())
}

