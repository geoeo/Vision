extern crate nalgebra as na;
extern crate num_traits;

use std::boxed::Box;
use std::ops::AddAssign;
use std::marker::{Send,Sync};
use num_traits::float;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use na::{convert, zero, DMatrix, DVector , OVector, Dyn, Matrix, SMatrix, SVector,Vector,Dim,storage::{Storage,StorageMut},base::{default_allocator::DefaultAllocator, allocator::Allocator},
    VecStorage, Const, DimMin, U1
};
use crate::numerics::{weighting::WeightingFunction, conjugate_gradient};
use crate::GenericFloat;

pub fn calc_weight_vec<F, D, S1>(
    residuals: &DVector<F>,
    std: Option<F>,
    weight_function: &Box<dyn WeightingFunction<F> + Send + Sync>,
    weights_vec: &mut Vector<F,D,S1>) -> () where 
        F : GenericFloat,
        D: Dim,
        S1: StorageMut<F, D>{
    for i in 0..residuals.len() {
        //Since sparse matricies are not integrated properly we weight by sqrt so that in matrix multiplication we get the correct value
        weights_vec[i] = float::Float::sqrt(weight_function.weight(residuals,i,std));
    }

}

pub fn calc_sqrt_weight_matrix<F>(
    residuals: &DVector<F>,
    std: Option<F>,
    weight_function: &Box<dyn WeightingFunction<F> + Send + Sync>) -> CsrMatrix<F> where 
        F : GenericFloat {
    let mut coo = CooMatrix::new(residuals.len(), residuals.len());
    for i in 0..residuals.len() {
        coo.push(i,i,float::Float::sqrt(weight_function.weight(residuals,i,std)));
    }
    CsrMatrix::from(&coo)
}

/**
 * The values in the weights vector should be the square root of the weight matrix diagonals
 * */
pub fn weight_residuals_sparse<F, D, S1,S2>(
    residual_target: &mut Vector<F,D,S1>,
     weights_vec: &Vector<F,D,S2>) -> () where 
        F : GenericFloat,
        D: Dim,
        S1: StorageMut<F, D>,
        S2: Storage<F, D> {
    residual_target.component_mul_assign(weights_vec);
}


//TODO: optimize
//TODO: performance offender
pub fn weight_jacobian_sparse<F, R,C,S1,S2>(
    jacobian: &mut Matrix<F, R, C, S1>,
    weights_vec: &Vector<F,R,S2>) -> () where
    F : GenericFloat,
    R: Dim,
    C: Dim ,
    S1: StorageMut<F, R,C> ,
    S2: Storage<F, R>,
    DefaultAllocator: Allocator<F, U1, C>
  {
    let size = weights_vec.len();
    for i in 0..size {
        let row = jacobian.row_mut(i) * weights_vec[i];
        jacobian.row_mut(i).copy_from(&row);
    }
}


pub fn scale_to_diagonal<F, const T: usize>(
    mat: &mut Matrix<F, Dyn, Const<T>, VecStorage<F, Dyn, Const<T>>>,
    residual: &DVector<F>,
    first_deriv: F,
    second_deriv: F,
) -> () where F : GenericFloat {
    for j in 0..T {
        for i in 0..residual.nrows() {
            mat[(i, j)] *= first_deriv + convert::<f64,F>(2.0) * second_deriv * float::Float::powi(residual[i], 2);
        }
    }

}

pub fn compute_cost<F>(residuals: &DVector<F>, std: Option<F>, weight_function: &Box<dyn WeightingFunction<F> + Send + Sync>) -> F where F : GenericFloat {
    weight_function.cost(residuals,std)
}

pub fn weight_residuals<F, const T: usize>(residual: &mut SVector<F, T>, weights: &SMatrix<F,T,T>) -> () where 
    F : GenericFloat,
    Const<T>: DimMin<Const<T>, Output = Const<T>> {
    weights.mul_to(&residual.clone(),residual);
}

pub fn weight_jacobian<F, const M: usize, const N: usize>(jacobian: &mut SMatrix<F,M,N>, weights: &SMatrix<F,M,M>) -> () 
    where 
    F : GenericFloat,
    Const<M>: DimMin<Const<M>, Output = Const<M>>,Const<N>: DimMin<Const<N>, Output = Const<N>> {
    weights.mul_to(&jacobian.clone(),jacobian);
}

#[allow(non_snake_case)]
pub fn gauss_newton_step_with_schur<F, R, C, S1, S2,StorageTargetArrow, StorageTargetResidual, const LANDMARK_PARAM_SIZE: usize, const CAMERA_PARAM_SIZE: usize>(
    target_arrowhead: &mut Matrix<F,C,C,StorageTargetArrow>, 
    target_arrowhead_residual: &mut Vector<F,C,StorageTargetResidual>, 
    target_perturb: &mut Vector<F,C,StorageTargetResidual>, 
    residuals: &Vector<F, R,S1>, 
    jacobian: &Matrix<F,R,C,S2>,
    mu: Option<F>, 
    tau: F,
    n_cams: usize,
    n_points: usize,
    u_span: usize, 
    v_span: usize)-> Option<(F,F)>
     where 
        F : GenericFloat,
        R: Dim, 
        C: Dim,
        S1: Storage<F, R>,
        S2: Storage<F, R, C>,
        StorageTargetArrow: StorageMut<F, C, C>,
        StorageTargetResidual: StorageMut<F, C, U1>,
        DefaultAllocator: Allocator<F, R, C>+  Allocator<F, C, R> + Allocator<F, C, C> + Allocator<F, C> + Allocator<F, U1, C> + Allocator<f64, U1, R>  {

        let mu_val = compute_arrow_head_and_residuals::<_,_,_,_,_,_,_,LANDMARK_PARAM_SIZE,CAMERA_PARAM_SIZE>(target_arrowhead,target_arrowhead_residual,jacobian,residuals,mu,tau,n_cams,n_points);

        /*
         *     | U*  W  |
         * H = | W_t V* |
         *  
         */

        let mut V_star_coo = CooMatrix::<F>::new(v_span, v_span);
        for r in u_span..(u_span+v_span) {
            for c in u_span..(u_span+v_span) {
                let v = target_arrowhead[(r,c)];
                if !v.is_zero() {
                    V_star_coo.push(r-u_span, c-u_span, v);
                }
            }
        }
        let V_star_csc = CsrMatrix::from(&V_star_coo);
        let mut V_star_inv_coo = CooMatrix::<F>::new(v_span, v_span);

        let mut inv_success = true;
        for i in (0..v_span).step_by(LANDMARK_PARAM_SIZE) {
            let mut v_slice = DMatrix::<F>::zeros(LANDMARK_PARAM_SIZE, LANDMARK_PARAM_SIZE);
            let slice_end = i+LANDMARK_PARAM_SIZE;
            for r in i..slice_end {
                for c in i..slice_end {
                    let entry = V_star_csc.get_entry(r, c).expect("V Star CSC indexing failed!");
                    let v = match entry {
                        nalgebra_sparse::SparseEntry::NonZero(v) => *v,
                        nalgebra_sparse::SparseEntry::Zero => F::zero()
                    };
                    v_slice[(r-i,c-i)] = v;
                }
            }
            let v_slice_cholesky = v_slice.cholesky();
            let success = match v_slice_cholesky {
                Some(chol) => {
                    let chol_inverse = chol.inverse();
                    for r in i..slice_end {
                        for c in i..slice_end {
                            let v = chol_inverse[(r-i,c-i)];
                            if !v.is_zero() {
                                V_star_inv_coo.push(r,c,v);
                            }
                        }
                    }
                    true
                },
                None => false
            };

            inv_success &= success;
        }

        match inv_success {
            true => {
                let V_star_inv_csc = CsrMatrix::from(&V_star_inv_coo);
                let U_star = target_arrowhead.view((0,0),(u_span,u_span));

                let res_a = target_arrowhead_residual.view((0,0),(u_span,1));
                let res_b = target_arrowhead_residual.view((u_span,0),(v_span,1)).into_owned();

                let W = target_arrowhead.view((0,u_span),(u_span,v_span));
                let W_t = target_arrowhead.view((u_span,0),(v_span,u_span)).into_owned();

                let schur_compliment = U_star -W*(&V_star_inv_csc*&W_t); // takes long time
                let res_a_augment = res_a-W*(&V_star_inv_csc*&res_b); // takes long time
        
                let h_a_option = schur_compliment.cholesky();
        
                match h_a_option {
                    Some(h_a_cholesky) => {
                        let h_a = h_a_cholesky.solve(&res_a_augment);
                        let h_b = (V_star_inv_csc)*(res_b-W_t*(&h_a));
        
                        target_perturb.view_mut((0,0),(u_span,1)).copy_from(&h_a);
                        target_perturb.view_mut((u_span,0),(v_span,1)).copy_from(&h_b);
                        
                        Some((compute_gain_ratio(target_perturb,target_arrowhead_residual,mu_val), mu_val))
                    }
                    _ => None
                }
            },
            false => None
        }

}

//TODO: This seems buggy: Investigate
#[allow(non_snake_case)]
pub fn gauss_newton_step_with_conguate_gradient<F, R, C, S1, S2,StorageTargetArrow, StorageTargetResidual, const LANDMARK_PARAM_SIZE: usize, const CAMERA_PARAM_SIZE: usize>(
    target_arrowhead: &mut Matrix<F,C,C,StorageTargetArrow>, 
    target_arrowhead_residual: &mut Vector<F,C,StorageTargetResidual>, 
    target_perturb: &mut Vector<F,C,StorageTargetResidual>, 
    V_star_inv: &mut DMatrix<F>,
    preconditioner: &mut DMatrix<F>,
    residuals: &Vector<F, R,S1>, 
    jacobian: &Matrix<F,R,C,S2>,
    mu: Option<F>, 
    tau: F,
    n_cams: usize,
    n_points: usize,
    u_span: usize, 
    v_span: usize,
    cg_tresh: F,
    cg_max_it: usize)-> Option<(F,F)>
     where 
        F : GenericFloat,
        R: Dim, 
        C: Dim,
        S1: Storage<F, R>,
        S2: Storage<F, R, C>,
        StorageTargetArrow: StorageMut<F, C, C>,
        StorageTargetResidual: StorageMut<F, C, U1>,
        DefaultAllocator: Allocator<F, R, C>+  Allocator<F, C, R> + Allocator<F, C, C> + Allocator<F, C> + Allocator<F, U1, C> + Allocator<F, U1, R>  {

        let mu_val = compute_arrow_head_and_residuals::<_,_,_,_,_,_,_,LANDMARK_PARAM_SIZE,CAMERA_PARAM_SIZE>(target_arrowhead,target_arrowhead_residual,jacobian,residuals,mu,tau,n_cams,n_points);

        /*
         *     | U*  W  |
         * H = | W_t V* |
         *  
         */

        let U_star = target_arrowhead.view((0,0),(u_span,u_span));
        let V_star = target_arrowhead.view((u_span,u_span),(v_span,v_span));
        
        let mut inv_success = true;
        for i in (0..v_span).step_by(LANDMARK_PARAM_SIZE) {
            let v_slice_cholesky = V_star.fixed_view::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(i,i).cholesky();
            let success = match v_slice_cholesky {
                Some(chol) => {
                    V_star_inv.fixed_view_mut::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(i,i).copy_from(&chol.inverse());
                    true
                },
                None => false
            };

            inv_success &= success;
        }

        let W = target_arrowhead.view((0,u_span),(u_span,v_span));
        let W_t = target_arrowhead.view((u_span,0),(v_span,u_span));
        let res_a = target_arrowhead_residual.rows(0, u_span);
        let res_b = target_arrowhead_residual.rows(u_span,v_span);

        let schur_compliment = U_star - W*(V_star_inv as &DMatrix<F>)*W_t; // takes long time

        for i in (0..u_span).step_by(CAMERA_PARAM_SIZE) {
            let s_cholesky =  schur_compliment.fixed_view::<CAMERA_PARAM_SIZE,CAMERA_PARAM_SIZE>(i,i).cholesky();
            let success = match s_cholesky {
                Some(chol) => {
                    preconditioner.fixed_view_mut::<CAMERA_PARAM_SIZE,CAMERA_PARAM_SIZE>(i,i).copy_from(&chol.inverse());
                    true
                },
                None => false
            };
            inv_success &= success;
        }

        match inv_success {
            true => {
                let schur_compliment_preconditioned = (preconditioner as &DMatrix<F>)*schur_compliment;
                let res_a_augment = (preconditioner as &DMatrix<F>)*(res_a-W*(V_star_inv as &DMatrix<F>)*res_b); // takes long time
                match conjugate_gradient::conjugate_gradient::<_,_,_,_,Dyn>(&schur_compliment_preconditioned, &res_a_augment, &mut target_perturb.rows_mut(0,u_span), cg_tresh, cg_max_it) {
                    true => {
                        let h_b = (V_star_inv as &DMatrix<F>)*(res_b-W_t*(&target_perturb.rows(0,u_span)));
                        target_perturb.view_mut((u_span,0),(v_span,1)).copy_from(&h_b);
                        Some((compute_gain_ratio(target_perturb,target_arrowhead_residual,mu_val), mu_val))
                    },
                    false => None
                }
            },
            false => None
        }



}

pub fn compute_gain_ratio<F,St, C>(perturb: &Vector<F,C, St>,residual: &Vector<F,C,St> , mu: F) -> F 
    where 
        F : GenericFloat,
        C: Dim,
        St: StorageMut<F, C, U1>,
        DefaultAllocator: Allocator<F, C> + Allocator<F, U1, C> {
    let scaled_target_res = perturb.scale(mu);
    let g_1 = perturb.transpose()*(&scaled_target_res);
    let g_2 = perturb.transpose()*residual;
    let gain_ratio_denom = convert::<f64,F>(0.5)*(g_1[0]+g_2[0]);
    gain_ratio_denom
}

#[allow(non_snake_case)]
pub fn gauss_newton_step<F, R, C,S1, S2, S3>(
    residuals: &Vector<F, R,S1>, 
    jacobian: &Matrix<F,R,C,S2>,
    identity: &Matrix<F,C,C,S3>,
     mu: Option<F>, 
     tau: F)-> Option<(OVector<F,C>,OVector<F,C>,F,F)>
     where 
        F : GenericFloat,
        R: Dim, 
        C: Dim + DimMin<C, Output = C>,
        S1: Storage<F, R>,
        S2: Storage<F, R, C>,
        S3: Storage<F, C, C>,
        DefaultAllocator: Allocator<F, R, C> +  Allocator<F, C, R> + Allocator<F, C, C> + Allocator<F, C>+ Allocator<F, Const<1_usize>, C> + Allocator<(usize, usize), C>  {
    let (A,g) = (jacobian.transpose()*jacobian,jacobian.transpose()*residuals);
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };
    let decomp_option = (A+ identity*mu_val).cholesky();
    match decomp_option {
        Some(decomp) => {
            let h = decomp.solve(&(-(&g)));
            let half: F = convert(0.5);
            let gain_ratio_denom = (&h).transpose()*((&h*mu_val)-(&g))*half;
            Some((h,g,gain_ratio_denom[0], mu_val))
        },
        None => None
    }

}

#[allow(non_snake_case)]
fn compute_arrow_head_and_residuals<F,R, C,StorageTargetArrow, StorageTargetResidual, StorageJacobian, StorageResidual, const LANDMARK_PARAM_SIZE: usize, const CAM_PARAM_SIZE: usize>
    (
        target_arrowhead: &mut Matrix<F,C,C,StorageTargetArrow>, 
        target_residual: &mut Vector<F,C,StorageTargetResidual>, 
        jacobian: &Matrix<F,R,C,StorageJacobian>, 
        residuals: &Vector<F, R, StorageResidual>,
        mu: Option<F>, 
        tau: F,
        n_cams: usize, 
        n_points: usize
    ) -> F
    where 
    F : GenericFloat,
    R: Dim, 
    C: Dim,
    StorageTargetArrow: StorageMut<F, C, C>,
    StorageTargetResidual: StorageMut<F, C, U1>,
    StorageResidual: Storage<F, R, U1>,
    StorageJacobian: Storage<F, R, C> {

        let number_of_cam_params = CAM_PARAM_SIZE*n_cams;
        let number_of_measurement_rows = 2*n_cams*n_points;
        let mut U_j = SMatrix::<F,CAM_PARAM_SIZE,CAM_PARAM_SIZE>::zeros();
        for j in (0..number_of_cam_params).step_by(CAM_PARAM_SIZE) {
            let u_idx = j;
            let cam_id = j/CAM_PARAM_SIZE;
            let row_start = 2*cam_id;
            let row_end = number_of_measurement_rows;
            for i in (row_start..row_end).step_by(2*n_cams){
                let feature_id = i/(2*n_cams);

                let slice_a = jacobian.fixed_view::<2,CAM_PARAM_SIZE>(i,j);
                let slice_a_transpose = slice_a.transpose();
                U_j += slice_a_transpose*slice_a;
                
                let v_idx = number_of_cam_params + feature_id*LANDMARK_PARAM_SIZE;
                let slice_b = jacobian.fixed_view::<2,LANDMARK_PARAM_SIZE>(i,v_idx);
                let slice_b_transpose = slice_b.transpose();
                
                let V_i = target_arrowhead.fixed_view::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(v_idx,v_idx)+ slice_b_transpose*slice_b;
                target_arrowhead.fixed_view_mut::<LANDMARK_PARAM_SIZE,LANDMARK_PARAM_SIZE>(v_idx,v_idx).copy_from(&V_i);

                let W_j = slice_a_transpose*slice_b;
                let W_j_transpose = W_j.transpose();

                target_arrowhead.fixed_view_mut::<CAM_PARAM_SIZE,CAM_PARAM_SIZE>(j,j).copy_from(&U_j);
                target_arrowhead.fixed_view_mut::<CAM_PARAM_SIZE,LANDMARK_PARAM_SIZE>(u_idx,v_idx).copy_from(&W_j);
                target_arrowhead.fixed_view_mut::<LANDMARK_PARAM_SIZE,CAM_PARAM_SIZE>(v_idx,u_idx).copy_from(&W_j_transpose);

                let residual = -residuals.fixed_view::<2,1>(i,0);
                target_residual.fixed_view_mut::<CAM_PARAM_SIZE,1>(u_idx,0).add_assign(&(slice_a_transpose*residual));
                target_residual.fixed_view_mut::<LANDMARK_PARAM_SIZE,1>(v_idx,0).add_assign(&(slice_b_transpose*residual));

            }
            target_arrowhead.fixed_view_mut::<CAM_PARAM_SIZE,CAM_PARAM_SIZE>(u_idx,u_idx).copy_from(&U_j);
            U_j.fill(zero::<F>());
        }

        let mut diag_max: F = zero::<F>();
        for i in 0..target_arrowhead.ncols() {
            let v = target_arrowhead[(i,i)];
            if v > diag_max {
                diag_max = v;
            }
        }

        let mu_val = match mu {
            None => tau*diag_max,
            Some(v) => v
        };

        for i in 0..target_arrowhead.ncols() {
            target_arrowhead[(i,i)] = target_arrowhead[(i,i)]+mu_val;
        }
        mu_val
}
