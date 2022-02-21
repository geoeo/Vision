extern crate nalgebra as na;

use na::{ DMatrix, DVector , OVector, Dynamic, Matrix, SMatrix, SVector,Vector,Dim,storage::{Storage,StorageMut},base::{default_allocator::DefaultAllocator, allocator::Allocator},
    VecStorage, Const, DimMin, U1
};

use std::boxed::Box;

use crate::numerics::{loss::LossFunction, weighting::WeightingFunction};
use crate::Float;


pub fn calc_weight_vec<D, S1>(
    residuals: &DVector<Float>,
    weight_function: &Box<dyn WeightingFunction>,
    weights_vec: &mut Vector<Float,D,S1>) -> () where 
        D: Dim,
        S1: StorageMut<Float, D>{
    let variance = weight_function.estimate_variance(residuals);
    for i in 0..residuals.len() {
        weights_vec[i] = weight_function.weight(residuals,i,variance);


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

#[allow(non_snake_case)]
pub fn gauss_newton_step_with_schur<R, C,S1, S2,StorageTargetArrow,StorageTargetResidual,const LANDMARK_PARAM_SIZE: usize, const CAMERA_PARAM_SIZE: usize>(
    target_arrowhead: &mut Matrix<Float,C,C,StorageTargetArrow>, 
    target_arrowhead_residual: &mut Vector<Float,C,StorageTargetResidual>, 
    target_perturb: &mut Vector<Float,C,StorageTargetResidual>, 
    residuals: &Vector<Float, R,S1>, 
    jacobian: &Matrix<Float,R,C,S2>,
    mu: Option<Float>, 
    tau: Float,
    n_cams: usize, 
    n_points: usize)-> (Float,Float) 
     where 
        R: Dim, 
        C: Dim + DimMin<C, Output = C>,
        S1: Storage<Float, R>,
        S2: Storage<Float, R, C>,
        StorageTargetArrow: StorageMut<Float, C, C>,
        StorageTargetResidual: StorageMut<Float, C, U1>,
        DefaultAllocator: Allocator<Float, R, C>+  Allocator<Float, C, R> + Allocator<Float, C, C> + Allocator<Float, C> + Allocator<Float, Const<1_usize>, C>  {


        let u_span = CAMERA_PARAM_SIZE*n_cams;
        let v_span = LANDMARK_PARAM_SIZE*n_points;

        let diag_max = compute_arrow_head_and_residuals::<_,_,_,_,_,_,LANDMARK_PARAM_SIZE,CAMERA_PARAM_SIZE>(target_arrowhead,target_arrowhead_residual,jacobian,residuals,mu,tau,n_cams,n_points);

        let U_star = target_arrowhead.slice((0,0),(u_span,u_span));
        let V_star = target_arrowhead.slice((u_span,u_span),(v_span,v_span));

        //TODO: preallocate this
        let mut V_star_inv = DMatrix::<Float>::zeros(v_span,v_span);
        for i in (0..v_span).step_by(LANDMARK_PARAM_SIZE) {
            let local_inv = V_star.fixed_slice::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(i,i).try_inverse().expect("local inverse failed");
            V_star_inv.fixed_slice_mut::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(i,i).copy_from(&local_inv);
        }
        let W = target_arrowhead.slice((0,u_span),(u_span,v_span));
        let W_t = target_arrowhead.slice((u_span,0),(v_span,u_span));
        let V_star_cholesky = V_star.cholesky().expect("v_star cholesky failed");
        

        let res_a = target_arrowhead_residual.slice((0,0),(u_span,1));
        let res_b = target_arrowhead_residual.slice((u_span,0),(v_span,1));


        let schur_compliment = U_star - W*(&V_star_inv)*W_t;
        let res_a_augment = res_a-W*V_star_inv*res_b;
        let h_a = schur_compliment.cholesky().expect("h_a schur cholesky failed").solve(&res_a_augment);
        let h_b = V_star_cholesky.solve(&(res_b-W_t*(&h_a)));


        target_perturb.slice_mut((0,0),(u_span,1)).copy_from(&h_a);
        target_perturb.slice_mut((u_span,0),(v_span,1)).copy_from(&h_b);

    
        let mu_val = match mu {
            None => tau*diag_max,
            Some(v) => v
        };
        let scaled_target_res = target_perturb.scale(mu_val);



        let g_1 = (&target_perturb).transpose()*(&scaled_target_res);
        let g_2 = (&target_perturb).transpose()*(target_arrowhead_residual as  &Vector<Float,C,StorageTargetResidual>);
        let gain_ratio_denom = g_1+g_2;
        
        (gain_ratio_denom[0], mu_val)


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
        DefaultAllocator: Allocator<Float, R, C>+  Allocator<Float, C, R> + Allocator<Float, C, C> + Allocator<Float, C>+ Allocator<Float, Const<1_usize>, C> + Allocator<(usize, usize), C>,  {
    let (A,g) = (jacobian.transpose()*jacobian,jacobian.transpose()*residuals);
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };
    let decomp = (A+ mu_val*identity).qr();
    let h = decomp.solve(&(-(&g))).expect("QR Solve Failed");
    let gain_ratio_denom = 0.5*(&h).transpose()*(mu_val*(&h)-(&g));
    (h,g,gain_ratio_denom[0], mu_val)
}

#[allow(non_snake_case)]
fn compute_arrow_head_and_residuals<R, C,StorageTargetArrow, StorageTargetResidual, StorageJacobian, StorageResidual, const LANDMARK_PARAM_SIZE: usize, const CAM_PARAM_SIZE: usize>
    (
        target_arrowhead: &mut Matrix<Float,C,C,StorageTargetArrow>, 
        target_residual: &mut Vector<Float,C,StorageTargetResidual>, 
        jacobian: &Matrix<Float,R,C,StorageJacobian>, 
        residuals: &Vector<Float, R, StorageResidual>,
        mu: Option<Float>, 
        tau: Float,
        n_cams: usize, 
        n_points: usize
    ) -> Float
    where 
    R: Dim, 
    C: Dim + DimMin<C, Output = C>,
    StorageTargetArrow: StorageMut<Float, C, C>,
    StorageTargetResidual: StorageMut<Float, C, U1>,
    StorageResidual: Storage<Float, R, U1>,
    StorageJacobian: Storage<Float, R, C> {

        let number_of_cam_params = CAM_PARAM_SIZE*n_cams;
        let number_of_measurement_rows = 2*n_cams*n_points;
        let mut diag_max: Float = 0.0;
    
        for j in (0..number_of_cam_params).step_by(CAM_PARAM_SIZE) {
            let mut U_j = SMatrix::<Float,CAM_PARAM_SIZE,CAM_PARAM_SIZE>::zeros();
            let u_idx = j;

            let cam_id = j/CAM_PARAM_SIZE;
            let row_start = 2*cam_id;
            let row_end = number_of_measurement_rows;
            for i in (row_start..row_end).step_by(2*n_cams){
                let feature_id = i/(2*n_cams);

                let slice_a = jacobian.fixed_slice::<2,CAM_PARAM_SIZE>(i,j);
                let slice_a_transpose = slice_a.transpose();
                U_j += slice_a_transpose*slice_a;
                
                let v_idx = number_of_cam_params + feature_id*LANDMARK_PARAM_SIZE;
                let slice_b = jacobian.fixed_slice::<2,LANDMARK_PARAM_SIZE>(i,v_idx);
                let slice_b_transpose = slice_b.transpose();
                
                let V_i = target_arrowhead.fixed_slice::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(v_idx,v_idx)+ slice_b_transpose*slice_b;
                target_arrowhead.fixed_slice_mut::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(v_idx,v_idx).copy_from(&V_i);

                let W_j = slice_a_transpose*slice_b;
                let W_j_transpose = W_j.transpose();

                target_arrowhead.fixed_slice_mut::<CAM_PARAM_SIZE,CAM_PARAM_SIZE>(j,j).copy_from(&U_j);
                target_arrowhead.fixed_slice_mut::<CAM_PARAM_SIZE,LANDMARK_PARAM_SIZE>(u_idx,v_idx).copy_from(&W_j);
                target_arrowhead.fixed_slice_mut::<LANDMARK_PARAM_SIZE,CAM_PARAM_SIZE>(v_idx,u_idx).copy_from(&W_j_transpose);

                let residual = -residuals.fixed_slice::<2,1>(i,0);
                let e_a = target_residual.fixed_slice::<CAM_PARAM_SIZE,1>(u_idx,0) + slice_a_transpose*residual;
                let e_b = target_residual.fixed_slice::<LANDMARK_PARAM_SIZE,1>(v_idx,0) + slice_b_transpose*residual;

                target_residual.fixed_slice_mut::<CAM_PARAM_SIZE,1>(u_idx,0).copy_from(&e_a);
                target_residual.fixed_slice_mut::<LANDMARK_PARAM_SIZE,1>(v_idx,0).copy_from(&e_b);

                let v_diag_max = V_i.diagonal().max();
                if v_diag_max > diag_max {
                    diag_max = v_diag_max;
                }

            }
            target_arrowhead.fixed_slice_mut::<CAM_PARAM_SIZE,CAM_PARAM_SIZE>(u_idx,u_idx).copy_from(&U_j);
            let u_diag_max = U_j.diagonal().max();
            if u_diag_max > diag_max {
                diag_max = u_diag_max;
            }
        }

        let mu_val = match mu {
            None => tau*diag_max,
            Some(v) => v
        };

        for i in 0..target_arrowhead.ncols() {
            target_arrowhead[(i,i)] = target_arrowhead[(i,i)]+mu_val;
        }
    diag_max
}
