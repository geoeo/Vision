extern crate nalgebra as na;

use na::{ DMatrix, DVector,OVector, Dynamic, Matrix, SMatrix, SVector,Vector,Dim,storage::{Storage,StorageMut},base::{default_allocator::DefaultAllocator, allocator::Allocator},
    VecStorage, Const, DimMin, U1
};
use std::boxed::Box;

use crate::numerics::{loss::LossFunction, weighting::WeightingFunction};
use crate::{float, Float};




pub fn norm<D, S1,S2>(
    residuals: &Vector<Float,D,S2>,
    weight_function: &Box<dyn WeightingFunction>,
    weights_vec: &mut Vector<Float,D,S1>) -> () where 
        D: Dim,
        S1: StorageMut<Float, D>,
        S2: Storage<Float, D>{
    for i in 0..residuals.len() {
        let res = residuals[i];
        weights_vec[i] = weight_function.cost(res).sqrt();


    }
}

/**
 * The values in the weights vector should be the square root of the weight matrix diagonals
 * */
pub fn weight_residuals_sparse<D, S1,S2>(
    residual_target: &mut Vector<Float,D,S1>,
     weights_vec: &Vector<Float,D,S2>) -> () where 
        D: Dim,
        S1: StorageMut<Float, D>,
        S2: Storage<Float, D> {
    residual_target.component_mul_assign(weights_vec);
}


//TODO: optimize
//TODO: performance offender
pub fn weight_jacobian_sparse<R,C,S1,S2>(
    jacobian: &mut Matrix<Float, R, C, S1>,
    weights_vec: &Vector<Float,R,S2>,) -> () where
    R: Dim,
    C: Dim ,
    S1: StorageMut<Float, R,C> ,
    S2: Storage<Float, R>,
    DefaultAllocator: Allocator<Float, U1, C>
  {
    let size = weights_vec.len();
    for i in 0..size {
        let row = jacobian.row_mut(i) * weights_vec[i];
        jacobian.row_mut(i).copy_from(&row);
    }
}


pub fn scale_to_diagonal<const T: usize>(
    mat: &mut Matrix<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>,
    residual: &DVector<Float>,
    first_deriv: Float,
    second_deriv: Float,
) -> () {
    for j in 0..T {
        for i in 0..residual.nrows() {
            mat[(i, j)] *= first_deriv + 2.0 * second_deriv * residual[i].powi(2);
        }
    }

}


pub fn compute_cost(residuals: &DVector<Float>, loss_function: &Box<dyn LossFunction>) -> Float {
    loss_function.cost((residuals.transpose() * residuals)[0])
}

pub fn weight_residuals<const T: usize>(residual: &mut SVector<Float, T>, weights: &SMatrix<Float,T,T>) -> () where Const<T>: DimMin<Const<T>, Output = Const<T>> {
    weights.mul_to(&residual.clone(),residual);
}

pub fn weight_jacobian<const M: usize, const N: usize>(jacobian: &mut SMatrix<Float,M,N>, weights: &SMatrix<Float,M,M>) -> () 
    where Const<M>: DimMin<Const<M>, Output = Const<M>>,Const<N>: DimMin<Const<N>, Output = Const<N>> {
    weights.mul_to(&jacobian.clone(),jacobian);
}

//TODO: use schur compliment
//TODO: only schur version
#[allow(non_snake_case)]
pub fn gauss_newton_step_with_loss_and_schur(
    residuals: &DVector<Float>,
    jacobian: &DMatrix<Float>,
    identity: &DMatrix<Float>,
    mu: Option<Float>,
    tau: Float,
    current_cost: Float,
    loss_function: &Box<dyn LossFunction>,
    rescaled_jacobian_target: &mut DMatrix<Float>,
    rescaled_residuals_target: &mut DVector<Float>
) -> (
    DVector<Float>,
    DVector<Float>,
    Float,
    Float
) {
    let selected_root = loss_function.select_root(current_cost);
    let first_deriv_at_cost = loss_function.first_derivative_at_current(current_cost);
    let second_deriv_at_cost = loss_function.second_derivative_at_current(current_cost);
    let is_curvature_negative = second_deriv_at_cost * current_cost < -0.5 * first_deriv_at_cost;

    let (A, g) = match selected_root {
        root if root != 0.0 => match is_curvature_negative {
            false => {
                let first_derivative_sqrt = first_deriv_at_cost.sqrt();
                let jacobian_factor = selected_root / current_cost;
                let residual_scale = first_derivative_sqrt / (1.0 - selected_root);
                let res_j = residuals.transpose() * jacobian;
                for i in 0..jacobian.nrows() {
                    rescaled_jacobian_target.row_mut(i).copy_from(
                        &(first_derivative_sqrt
                            * (jacobian.row(i) - (jacobian_factor * residuals[i] * (&res_j)))),
                    );
                    rescaled_residuals_target[i] = residual_scale * residuals[i];
                }
                (
                    rescaled_jacobian_target.transpose()
                        * rescaled_jacobian_target
                            as &DMatrix<Float>,
                    rescaled_jacobian_target.transpose()
                        * rescaled_residuals_target as &DVector<Float>,
                )
            }
            _ => {
                (jacobian.transpose()*first_deriv_at_cost*jacobian+2.0*second_deriv_at_cost*jacobian.transpose() * residuals*residuals.transpose() * jacobian,
                first_deriv_at_cost * jacobian.transpose() * residuals)
            }
        },
        _ => (
            jacobian.transpose() * jacobian,
            jacobian.transpose() * residuals,
        ),
    };
    let mu_val = match mu {
        None => tau * A.diagonal().max(),
        Some(v) => v,
    };

    let decomp = (A + mu_val * identity).qr();
    let h = decomp.solve(&(-(&g))).expect("QR Solve Failed");
    let gain_ratio_denom = (&h).transpose() * (mu_val * (&h) - (&g));
    (h, g, gain_ratio_denom[0], mu_val)
}

//TODO: use schur compliment
#[allow(non_snake_case)]
pub fn gauss_newton_step_with_schur<R, C,S1, S2, S3>(
    residuals: &Vector<Float, R,S1>, 
    jacobian: &Matrix<Float,R,C,S2>,
    identity: &Matrix<Float,C,C,S3>,
     mu: Option<Float>, 
     tau: Float)-> (OVector<Float,C>,OVector<Float,C>,Float,Float) 
     where 
        R: Dim, 
        C: Dim + DimMin<C, Output = C>,
        S1: Storage<Float, R>,
        S2: Storage<Float, R, C>,
        S3: Storage<Float, C, C>,
        DefaultAllocator: Allocator<Float, R, C>+  Allocator<Float, C, R> + Allocator<Float, C, C> + Allocator<Float, C>+ Allocator<Float, Const<1_usize>, C>  {
    let (A,g) = (jacobian.transpose()*jacobian,jacobian.transpose()*residuals);
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let decomp = (A+ mu_val*identity).qr();
    let h = decomp.solve(&(-(&g))).expect("QR Solve Failed");
    let gain_ratio_denom = (&h).transpose()*(mu_val*(&h)-(&g));
    (h,g,gain_ratio_denom[0], mu_val)
}

#[allow(non_snake_case)]
pub fn gauss_newton_step<R, C,S1, S2, S3>(
    residuals: &Vector<Float, R,S1>, 
    jacobian: &Matrix<Float,R,C,S2>,
    identity: &Matrix<Float,C,C,S3>,
     mu: Option<Float>, 
     tau: Float)-> (OVector<Float,C>,OVector<Float,C>,Float,Float) 
     where 
        R: Dim, 
        C: Dim + DimMin<C, Output = C>,
        S1: Storage<Float, R>,
        S2: Storage<Float, R, C>,
        S3: Storage<Float, C, C>,
        DefaultAllocator: Allocator<Float, R, C>+  Allocator<Float, C, R> + Allocator<Float, C, C> + Allocator<Float, C>+ Allocator<Float, Const<1_usize>, C>  {
    let (A,g) = (jacobian.transpose()*jacobian,jacobian.transpose()*residuals);
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let decomp = (A+ mu_val*identity).qr();
    let h = decomp.solve(&(-(&g))).expect("QR Solve Failed");
    let gain_ratio_denom = (&h).transpose()*(mu_val*(&h)-(&g));
    (h,g,gain_ratio_denom[0], mu_val)
}
