extern crate nalgebra as na;

use na::{
    storage::Storage, DMatrix, DVector, Dynamic, Matrix, Matrix4, RowVector2, SMatrix, SVector,
    VecStorage, Vector3, Vector4, Vector6, U2, U4, U6, U9, Const, DimMin, DimMinimum
};
use std::boxed::Box;

use crate::numerics::{lie, loss::LossFunction};
use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::{
    imu_odometry,
    imu_odometry::imu_delta::ImuDelta,
    imu_odometry::{weight_residuals, weight_jacobian},
};
use crate::{float, Float};


//TODO: refactor this to work with arbirary number of elements

#[allow(non_snake_case)]
fn gauss_newton_step_with_loss<const T: usize>(
    residuals: &DVector<Float>,
    imu_residuals: &SVector<Float, T>,
    jacobian: &Matrix<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>,
    imu_jacobian: &SMatrix<Float,T,T>,
    identity: &SMatrix<Float,T,T>,
    mu: Option<Float>,
    tau: Float,
    current_cost: Float,
    loss_function: &Box<dyn LossFunction>,
    rescaled_jacobian_target: &mut Matrix<
        Float,
        Dynamic,
        Const<T>,
        VecStorage<Float, Dynamic, Const<T>>,
    >,
    rescaled_residuals_target: &mut DVector<Float>,
    imu_rescaled_jacobian_target: &mut SMatrix<Float, T,T>,
    imu_rescaled_residuals_target: &mut SVector<Float, T>,
) -> (
    SVector<Float, T>,
    SVector<Float, T>,
    Float,
    Float,
)  where Const<T>: DimMin<Const<T>, Output = Const<T>> {
    panic!("TODO")
}

pub fn norm(
    residuals: &DVector<Float>,
    loss_function: &Box<dyn LossFunction>,
    weights_vec: &mut DVector<Float>,
) -> () {
    for i in 0..residuals.len() {
        let res = residuals[i];
        weights_vec[i] = (loss_function.second_derivative_at_current(res) * res)
            .abs()
            .sqrt();
    }
}

pub fn weight_residuals_sparse(
    residual_target: &mut DVector<Float>,
    weights_vec: &DVector<Float>,
) -> () {
    residual_target.component_mul_assign(weights_vec);
}

//TODO: optimize
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