extern crate nalgebra as na;

use na::{ DMatrix, DVector, Dynamic, Matrix, SMatrix, SVector,
    VecStorage, Const, DimMin
};
use std::boxed::Box;

use crate::numerics::{lie, loss::LossFunction, weighting::WeightingFunction};
use crate::{float, Float};




pub fn norm(
    residuals: &DVector<Float>,
    weight_function: &Box<dyn WeightingFunction>,
    weights_vec: &mut DVector<Float>,
) -> () {
    for i in 0..residuals.len() {
        let res = residuals[i];
        weights_vec[i] = weight_function.cost(res).sqrt();


    }
}

/**
 * The values in the weights vector should be the square root of the weight matrix diagonals
 * */
pub fn weight_residuals_sparse(
    residual_target: &mut DVector<Float>,
    weights_vec: &DVector<Float>,
) -> () {
    residual_target.component_mul_assign(weights_vec);
}


//TODO: optimize
//TODO: performance offender
pub fn weight_jacobian_sparse<const T: usize>(
    jacobian: &mut Matrix<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>,
    weights_vec: &DVector<Float>,
) -> () {
    let size = weights_vec.len();
    for i in 0..size {
        let weighted_row = jacobian.row(i) * weights_vec[i];
        jacobian.row_mut(i).copy_from(&weighted_row);
    }
}

pub fn weight_jacobian_sparse_dynamic(
    jacobian: &mut DMatrix<Float>,
    weights_vec: &DVector<Float>,
) -> () {
    let size = weights_vec.len();
    for i in 0..size {
        let weighted_row = jacobian.row(i) * weights_vec[i];
        jacobian.row_mut(i).copy_from(&weighted_row);
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

//TODO: optimize result matrices
#[allow(non_snake_case)]
pub fn gauss_newton_step_with_loss(
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
                (jacobian.transpose()*first_deriv_at_cost*jacobian+2.0*second_deriv_at_cost*jacobian.transpose() * residuals*residuals.transpose() * jacobian,first_deriv_at_cost * jacobian.transpose() * residuals)
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