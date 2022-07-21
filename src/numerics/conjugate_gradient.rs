extern crate nalgebra as na;

use std::ops::{AddAssign,SubAssign};
use na::{DMatrix, Matrix, storage::{Storage, StorageMut}, Vector, Const ,U1, Dim, base::{dimension::DimName,default_allocator::DefaultAllocator, allocator::{Allocator}}};
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

#[allow(non_snake_case)]
pub fn conjugate_gradient<StorageA, StorageB, StorageX, const STATE: usize>(A: &Matrix<Float,Const<STATE>,Const<STATE>,StorageA>,  b: &Vector<Float,Const<STATE>,StorageB>, x: &mut Vector<Float, Const<STATE>, StorageX>, threshold: Float, max_it: usize) -> () 
    where 
        StorageA: Storage<Float, Const<STATE>, Const<STATE>>, 
        StorageX: StorageMut<Float, Const<STATE>, U1>,
        StorageB: Storage<Float, Const<STATE>, U1>,
        DefaultAllocator: Allocator<Float, Const<STATE>, Const<STATE>> + Allocator<Float, Const<STATE>> + Allocator<Float, U1, Const<STATE>>  {

        let mut s = b - A*(x as &Vector<Float, Const<STATE>, StorageX>);
        let mut p = s.clone();
        let mut it = 0;

        while it < max_it && (&s).norm() > threshold {
            let p_borrow = &p;
            let s_comp_squared = ((&s).transpose()*(&s))[0];
            let p_comp_squared = (p_borrow.transpose()*A*p_borrow)[0];
            let alpha = s_comp_squared/p_comp_squared;
            x.add_assign(&(alpha*p_borrow));
            s.sub_assign(&(alpha*(A*p_borrow)));
            let s_new_comp_squared = ((&s).transpose()*(&s))[0];
            let beta = s_new_comp_squared/s_comp_squared;
            let p_new = &s + beta*p_borrow;
            p.fixed_rows_mut::<STATE>(0).copy_from(&p_new);
            it = it+1;
        }

}