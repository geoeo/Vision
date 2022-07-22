extern crate nalgebra as na;

use std::ops::{AddAssign,SubAssign};
use na::{DMatrix, Matrix,Dynamic, storage::{Storage, StorageMut}, Vector, U1, Dim, DimName, base::{default_allocator::DefaultAllocator, allocator::{Allocator}}};
use crate::{Float};

#[allow(non_snake_case)]
pub fn compute_preconditioner_inverse<PStorage,VStorage,WStorage, WtStorage>(preconditioner: &mut DMatrix::<Float>,P_inv: &Matrix::<Float,Dynamic,Dynamic,PStorage> , V_star_inv: &Matrix::<Float,Dynamic,Dynamic, VStorage>, W: &Matrix::<Float,Dynamic,Dynamic, WStorage>, W_t: &Matrix::<Float,Dynamic,Dynamic, WtStorage>, omega: Float) -> () 
    where
        PStorage: Storage<Float,Dynamic,Dynamic>,
        VStorage: Storage<Float,Dynamic,Dynamic> + Clone,
        WStorage: Storage<Float,Dynamic,Dynamic>,
        WtStorage: Storage<Float,Dynamic,Dynamic>,
    {
    let p_dim = P_inv.nrows();
    let v_dim = V_star_inv.nrows();

    let omega_sqrd = omega.powi(2);
    preconditioner.slice_mut((0,0),(p_dim,p_dim)).copy_from(P_inv);
    
    let temp = W*V_star_inv;
    preconditioner.slice_mut((0,p_dim),(p_dim,v_dim)).copy_from(&(P_inv*(&temp)));
    preconditioner.slice_mut((0,p_dim),(p_dim,v_dim)).scale_mut(-omega);
    
    let temp_2 = V_star_inv*W_t*P_inv;
    preconditioner.slice_mut((p_dim,0),(v_dim,p_dim)).copy_from(&temp_2);
    preconditioner.slice_mut((p_dim,0),(v_dim,p_dim)).scale_mut(-omega);

    preconditioner.slice_mut((p_dim,p_dim),(v_dim,v_dim)).copy_from(&((&temp_2)*(&temp)));
    preconditioner.slice_mut((p_dim,p_dim),(v_dim,v_dim)).scale_mut(omega_sqrd);
    preconditioner.slice_mut((p_dim,p_dim),(v_dim,v_dim)).add_assign(V_star_inv);


}

#[allow(non_snake_case)]
pub fn conjugate_gradient<StorageA, StorageB, StorageX, S>(A: &Matrix<Float,S,S,StorageA>,  b: &Vector<Float,S,StorageB>, x: &mut Vector<Float, S, StorageX>, threshold: Float, max_it: usize) -> () 
    where 
        S: Dim + DimName,
        StorageA: Storage<Float, S, S>, 
        StorageX: StorageMut<Float, S, U1>,
        StorageB: Storage<Float, S, U1>,
        DefaultAllocator: Allocator<Float, S, S> + Allocator<Float, S> + Allocator<Float, U1, S>  {

        let mut s = b - A*(x as &Vector<Float, S, StorageX>);
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
            p.rows_mut(0,p.nrows()).copy_from(&p_new);
            it = it+1;
        }

}