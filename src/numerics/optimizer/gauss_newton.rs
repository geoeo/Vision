extern crate nalgebra as na;
extern crate num_traits;


use std::marker::{Send,Sync};
use std::sync::mpsc;
use na::{DVector,DMatrix, base::Scalar, RealField, convert};
use simba::scalar::SupersetOf;
use num_traits::float;

use crate::sensors::camera::Camera;
use crate::numerics::{max_norm, least_squares::{compute_cost,weight_jacobian_sparse,weight_residuals_sparse, calc_weight_vec, gauss_newton_step}};
use crate::sfm::{landmark::Landmark,state::State};
use crate::sfm::runtime_parameters::RuntimeParameters; 
use crate::Float;

const CAMERA_PARAM_SIZE: usize = 6; //TODO make this generic with state


pub struct OptimizerGn<F: SupersetOf<Float>, C : Camera<Float>, L: Landmark<F,LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync, const LANDMARK_PARAM_SIZE: usize> where F: float::Float + Scalar + RealField {
    pub get_estimated_features: Box<dyn Fn(&State<F,L,LANDMARK_PARAM_SIZE>, &Vec<&C>, &DVector<F>, &mut DVector<F>) -> ()>,
    pub compute_residual: Box<dyn Fn(&DVector<F>, &DVector<F>, &mut DVector<F>) -> ()>,
    pub compute_jacobian: Box<dyn Fn(&State<F,L,LANDMARK_PARAM_SIZE>, &Vec<&C>, &mut DMatrix<F>) -> ()>,
    pub compute_state_size: Box<dyn Fn(&State<F,L,LANDMARK_PARAM_SIZE>) -> usize>
}

impl<F: SupersetOf<Float>, C : Camera<Float>, L: Landmark<F,LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync, const LANDMARK_PARAM_SIZE: usize> OptimizerGn<F,C,L,LANDMARK_PARAM_SIZE> where F: float::Float + Scalar + RealField {
    
    pub fn new(
        get_estimated_features: Box<dyn Fn(&State<F,L,LANDMARK_PARAM_SIZE>, &Vec<&C>, &DVector<F>, &mut DVector<F>) -> ()>,
        compute_residual: Box<dyn Fn(&DVector<F>, &DVector<F>, &mut DVector<F>) -> ()>,
        compute_jacobian: Box<dyn Fn( &State<F,L,LANDMARK_PARAM_SIZE>, &Vec<&C>, &mut DMatrix<F>) -> ()>,
        compute_state_size: Box<dyn Fn(&State<F,L,LANDMARK_PARAM_SIZE>) -> usize>

    ) -> OptimizerGn<F,C,L,LANDMARK_PARAM_SIZE> {
        OptimizerGn {
            get_estimated_features,
            compute_residual,
            compute_jacobian,
            compute_state_size
        }
    }
    
    pub fn optimize(&self,
        state: &mut State<F,L,LANDMARK_PARAM_SIZE>, cameras: &Vec<&C>, observed_features: &DVector<F>, runtime_parameters: &RuntimeParameters<F>, abort_receiver: Option<&mpsc::Receiver<bool>>, done_transmission: Option<&mpsc::Sender<bool>>
    ) -> Option<Vec<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; LANDMARK_PARAM_SIZE]>)>> where F: float::Float + Scalar + RealField {
    
        let max_iterations = runtime_parameters.max_iterations[0];
        
        let state_size = (self.compute_state_size)(state);
        let identity = DMatrix::<F>::identity(state_size, state_size);
        let mut new_state = state.clone();
        let mut jacobian = DMatrix::<F>::zeros(observed_features.nrows(),state_size); // a lot of memory
        let mut residuals = DVector::<F>::zeros(observed_features.nrows());
        let mut new_residuals = DVector::<F>::zeros(observed_features.nrows());
        let mut estimated_features = DVector::<F>::zeros(observed_features.nrows());
        let mut new_estimated_features = DVector::<F>::zeros(observed_features.nrows());
        let mut weights_vec = DVector::<F>::from_element(observed_features.nrows(),F::one());
        
        let mut debug_state_list = match runtime_parameters.debug {
            true => Some(Vec::<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; LANDMARK_PARAM_SIZE]>)>::with_capacity(max_iterations)),
            false => None
        };
        //let mut preconditioner = DMatrix::<F>::zeros(u_span,u_span); // a lot of memory - maybe use sparse format
        let two : F = convert(2.0);

        println!("BA Memory Allocation Complete.");

        (self.get_estimated_features)(state, cameras,observed_features, &mut estimated_features);
        (self.compute_residual)(&estimated_features, observed_features, &mut residuals);
        (self.compute_jacobian)(&state,&cameras,&mut jacobian);

        //TODO: weight cam and features independently
        let mut std: Option<F> = runtime_parameters.intensity_weighting_function.estimate_standard_deviation(&residuals);
        if std.is_some() {
            calc_weight_vec(
                &residuals,
                std,
                &runtime_parameters.intensity_weighting_function,
                &mut weights_vec,
            );
            weight_residuals_sparse(&mut residuals, &weights_vec); 
            weight_jacobian_sparse(&mut jacobian, &weights_vec);
        }

        let mut max_norm_delta = float::Float::max_value();
        let mut delta_thresh = float::Float::min_value();
        let mut delta_norm = float::Float::max_value();
        let mut nu: F = two;
        let tau = runtime_parameters.taus[0];

        let mut mu: Option<F> = match runtime_parameters.lm {
            true => None,
            false => Some(F::zero())
        };
        let step = match runtime_parameters.lm {
            true => F::one(),
            false => runtime_parameters.step_sizes[0]
        };

        let mut cost = compute_cost(&residuals,&runtime_parameters.intensity_weighting_function);
        let mut iteration_count = 0;
        let mut run = true;
        while ((!runtime_parameters.lm && (float::Float::sqrt(cost) > runtime_parameters.eps[0])) || 
        (runtime_parameters.lm && delta_norm > delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps && float::Float::sqrt(cost) > runtime_parameters.eps[0] ))  && iteration_count < max_iterations && run {
            println!("it: {}, avg_rmse: {}",iteration_count,float::Float::sqrt(cost));
            if runtime_parameters.debug {
                debug_state_list.as_mut().expect("Debug is true but state list is None!. This should not happen").push(state.to_serial());
            }
            
            let (delta,g,gain_ratio_denom, mu_val) = gauss_newton_step::<_,_,_,_,_,_>(
                &residuals,
                &jacobian,
                &identity,mu,tau
                );

            mu = Some(mu_val);
            let pertb = delta.scale(step);
            new_state.update(&pertb);
    
            (self.get_estimated_features)(&new_state, cameras,observed_features, &mut new_estimated_features);
            (self.compute_residual)(&new_estimated_features, observed_features, &mut new_residuals);
            std = runtime_parameters.intensity_weighting_function.estimate_standard_deviation(&residuals);
            if std.is_some() {
                calc_weight_vec(
                    &new_residuals,
                    std,
                    &runtime_parameters.intensity_weighting_function,
                    &mut weights_vec,
                );
                weight_residuals_sparse(&mut new_residuals, &weights_vec);
            }
    
    
            let new_cost = compute_cost(&new_residuals,&runtime_parameters.intensity_weighting_function);
            let cost_diff = cost-new_cost;
            let gain_ratio = match gain_ratio_denom {
                v if v != F::zero() => cost_diff/v,
                _ => float::Float::nan()
            };
            (gain_ratio, new_cost, pertb.norm(), cost_diff);

            println!("cost: {}, new cost: {}, mu: {:?}, gain: {} , nu: {}, std: {:?}",cost,new_cost, mu, gain_ratio, nu, std);
            
            if (!gain_ratio.is_nan() && gain_ratio > F::zero() && cost_diff > F::zero()) || !runtime_parameters.lm {
                estimated_features.copy_from(&new_estimated_features);
                state.copy_from(&new_state); 

                cost = new_cost;

                max_norm_delta = max_norm(&g);
                delta_norm = pertb.norm(); 

                delta_thresh = runtime_parameters.delta_eps*(estimated_features.norm() + runtime_parameters.delta_eps);

                residuals.copy_from(&new_residuals);

                jacobian.fill(F::zero());
                (self.compute_jacobian)(&state,&cameras,&mut jacobian);
                if std.is_some() {
                    weight_jacobian_sparse(&mut jacobian, &weights_vec);
                }

                let v: F = convert(1.0 / 3.0);
                mu = Some(mu.unwrap() * float::Float::max(v,F::one() - float::Float::powi(two * gain_ratio - F::one(),3)));
                nu = two;
            } else {
                new_state.copy_from(&state); 
                mu = match mu {
                    Some(v) => Some(nu*v),
                    None => None
                };
                nu *= two;
            }

            iteration_count += 1;

            if (mu.is_some() && mu.unwrap().is_infinite()) || nu.is_infinite(){
                break;
            }

            if abort_receiver.is_some() {
                let rx = abort_receiver.unwrap();
                run = match rx.try_recv() {
                    Err(_) => true,
                    Ok(b) => !b
                };
            }
        }
        println!("Solver Converged: it: {}, avg_rmse: {}",iteration_count,float::Float::sqrt(cost));
        if done_transmission.is_some() {
            done_transmission.unwrap().send(true).unwrap();
        }
        
        debug_state_list
    }

}