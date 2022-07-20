extern crate nalgebra as na;

use na::{DMatrix};
use crate::{Float};

#[allow(non_snake_case)]
pub fn compute_preconditioner(preconditioner: &mut DMatrix::<Float>,P: DMatrix::<Float> , V_star: &DMatrix::<Float>, V_star_inv: &DMatrix::<Float>, W: &DMatrix::<Float>, W_t: &DMatrix::<Float>, omega: Float ) -> () {
    let p_rows = P.nrows();
    let p_cols = P.ncols();
    let w_rows = W.nrows();
    let w_cols = W.ncols();
    let v_rows = V_star.nrows();
    let v_cols = V_star.ncols();

    let omega_sqrd = omega.powi(2);
    preconditioner.slice_mut((0,0),(p_rows,p_cols)).copy_from(&(P + omega_sqrd*W*V_star_inv*W_t));
    preconditioner.slice_mut((0,p_cols),(w_rows,w_cols)).copy_from(&W);
    preconditioner.slice_mut((0,p_cols),(w_rows,w_cols)).scale_mut(omega);
    preconditioner.slice_mut((p_rows,0),(w_rows,w_cols)).copy_from(&W_t);
    preconditioner.slice_mut((p_rows,0),(w_rows,w_cols)).scale_mut(omega);
    preconditioner.slice_mut((p_rows,p_cols),(v_rows,v_cols)).copy_from(&V_star);




}