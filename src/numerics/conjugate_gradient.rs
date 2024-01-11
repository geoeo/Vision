extern crate nalgebra as na;
extern crate num_traits;

use std::ops::{AddAssign,SubAssign};
use num_traits::float;
use na::{DMatrix, Matrix,Dyn, storage::{Storage, StorageMut}, Vector ,U1, Dim, base::{default_allocator::DefaultAllocator, allocator::Allocator}};
use crate::GenericFloat;

#[allow(non_snake_case)]
pub fn compute_block_matrix_preconditioner_inverse<F,PStorage,VStorage,WStorage, WtStorage>(preconditioner: &mut DMatrix::<F>,P_inv: &Matrix::<F,Dyn,Dyn,PStorage> , C: &Matrix::<F,Dyn,Dyn, VStorage>, E: &Matrix::<F,Dyn,Dyn, WStorage>, E_t: &Matrix::<F,Dyn,Dyn, WtStorage>, omega: F) -> () 
    where
        F : GenericFloat,
        PStorage: Storage<F,Dyn,Dyn>,
        VStorage: Storage<F,Dyn,Dyn> + Clone,
        WStorage: Storage<F,Dyn,Dyn>,
        WtStorage: Storage<F,Dyn,Dyn> {
    let p_dim = P_inv.nrows();
    let v_dim = C.nrows();

    let omega_sqrd = float::Float::powi(omega, 2);
    preconditioner.view_mut((0,0),(p_dim,p_dim)).copy_from(P_inv);
    
    let temp = E*C;
    preconditioner.view_mut((0,p_dim),(p_dim,v_dim)).copy_from(&(P_inv*(&temp)));
    preconditioner.view_mut((0,p_dim),(p_dim,v_dim)).scale_mut(-omega);
    
    let temp_2 = C*E_t*P_inv;
    preconditioner.view_mut((p_dim,0),(v_dim,p_dim)).copy_from(&temp_2);
    preconditioner.view_mut((p_dim,0),(v_dim,p_dim)).scale_mut(-omega);

    preconditioner.view_mut((p_dim,p_dim),(v_dim,v_dim)).copy_from(&((&temp_2)*(&temp)));
    preconditioner.view_mut((p_dim,p_dim),(v_dim,v_dim)).scale_mut(omega_sqrd);
    preconditioner.view_mut((p_dim,p_dim),(v_dim,v_dim)).add_assign(C);
}

#[allow(non_snake_case)]
pub fn conjugate_gradient<F,StorageA, StorageB, StorageX, S>(A: &Matrix<F,S,S,StorageA>,  b: &Vector<F,S,StorageB>, x: &mut Vector<F, S, StorageX>, threshold: F, max_it: usize) -> bool 
    where 
        F : GenericFloat,
        S: Dim,
        StorageA: Storage<F, S, S>, 
        StorageX: StorageMut<F, S, U1>,
        StorageB: Storage<F, S, U1>,
        DefaultAllocator: Allocator<F, S, S> + Allocator<F, S> + Allocator<F, U1, S>  {
        let mut s = b - A*(x as &Vector<F, S, StorageX>);
        let mut p = s.clone();
        let p_rows = p.nrows();
        let mut it = 0;
        let mut norm = (&s).norm();

        while it < max_it && norm > threshold {
            let p_borrow = &p;
            let s_comp_squared = ((&s).transpose()*(&s))[0];
            let p_comp_squared = (p_borrow.transpose()*A*p_borrow)[0];
            let alpha: F = s_comp_squared/p_comp_squared;
            x.add_assign(&(p_borrow.scale(alpha)));
            s.sub_assign(&((A*p_borrow).scale(alpha)));
            let s_new_comp_squared = ((&s).transpose()*(&s))[0];
            let beta = s_new_comp_squared/s_comp_squared;
            let p_new = &s + p_borrow.scale(beta);
            p.rows_mut(0,p_rows).copy_from(&p_new.rows(0,p_rows));
            it = it+1;
            norm = (&s).norm();
        }

        println!("cg finished with it: {} and thesh: {}" , it, norm);
        !(norm.is_nan() || norm.is_infinite())

}