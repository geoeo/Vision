extern crate nalgebra as na;

use na::{DVector,DMatrix,Matrix, Dynamic, U4, VecStorage, Vector4, Matrix2x4, Matrix4};
use crate::{float,Float};
use crate::sensors::camera::Camera;
use crate::numerics::lie::{exp, left_jacobian_around_identity};
use crate::numerics::{max_norm_dynamic, solver::{compute_cost,weight_jacobian_sparse_dynamic,weight_residuals_sparse, norm, gauss_newton_step_with_loss_and_schur}};
use crate::image::bundle_adjustment::state::State;
use crate::odometry::runtime_parameters::RuntimeParameters; //TODO remove dependency on odometry module

/**
 * In the format [f1_cam1_x,f1_cam1_y,f2_cam1_x,f2_cam1_y,f3_cam1_x,f3_cam1_y,f1_cam2_x,f1_cam2_y,f2_cam2_x,...]
 * */
pub fn get_estimated_features<C : Camera>(state: &State, cameras: &Vec<&C>, estimated_features: &mut DVector<Float>) -> () {
    let n_cams = state.n_cams;
    let n_points = state.n_points;
    let estimated_state = &state.data;
    assert_eq!(estimated_features.nrows(),n_cams*n_points*2);
    for i in (0..n_cams).step_by(6){
        let u = estimated_state.fixed_rows::<3>(i*n_cams);
        let w = estimated_state.fixed_rows::<3>(i*n_cams+3);
        let pose = exp(&u,&w);
        let camera = cameras[i];

        let mut position_per_cam = Matrix::<Float,U4,Dynamic, VecStorage<Float,U4,Dynamic>>::from_element(n_points, 1.0);
        for j in (0..n_points).step_by(3) {
            position_per_cam.fixed_rows_mut::<3>(j).copy_from(&estimated_state.fixed_rows::<3>(n_cams+j));
        };
        let transformed_points = pose*position_per_cam;
        for j in 0..n_points {
            let estimated_feature = camera.project(&transformed_points.fixed_slice::<3,1>(0,j));            
            estimated_features[i*n_points+2*j] = estimated_feature.x;
            estimated_features[i*n_points+2*j+1] = estimated_feature.y;
        }

    }

}

pub fn compute_residual(estimated_features: &DVector<Float>, observed_features: &DVector<Float>, residual_vector: &mut DVector<Float>) -> () {
    assert_eq!(residual_vector.nrows(), estimated_features.nrows()/2);
    let diff = observed_features - estimated_features;
    let diff_squared = diff.component_mul(&diff);
    for i in 0..residual_vector.nrows(){
        residual_vector[i] = (diff_squared[2*i] + diff_squared[2*i+1]).sqrt();
    }
}

pub fn compute_jacobian_wrt_object_points<C : Camera>(state: &State, cameras: &Vec<&C>, transformation: &Matrix4<Float>, i: usize, j: usize, jacobian: &mut DMatrix<Float>) -> () {

    let point = &state.data.fixed_rows::<3>(i);
    let transformed_point = transformation*Vector4::<Float>::new(point[0],point[1],point[2],1.0);
    let projection_jacobian = cameras[j].get_jacobian_with_respect_to_position(&transformed_point.fixed_rows::<3>(0));
    let mut projection_jacobian_homogeneous = Matrix2x4::<Float>::zeros();
    projection_jacobian_homogeneous.fixed_slice_mut::<2,3>(0,0).copy_from(&projection_jacobian);

    let row_idx = ((i/3)*state.n_cams+j)*2;
    let jacobian_homogeneous = &(projection_jacobian_homogeneous*transformation);
    jacobian.fixed_slice_mut::<2,3>(row_idx,6*state.n_cams+i).copy_from(&jacobian_homogeneous.fixed_slice::<2,3>(0,0));


}

pub fn compute_jacobian_wrt_camera_parameters<C : Camera>(state: &State, cameras: &Vec<&C>, transformation: &Matrix4<Float>,i: usize, j: usize, jacobian: &mut DMatrix<Float>) -> () {

    let point = &state.data.fixed_rows::<3>(i);
    let transformed_point = transformation*Vector4::<Float>::new(point[0],point[1],point[2],1.0);
    let lie_jacobian = left_jacobian_around_identity(&transformed_point.fixed_rows::<3>(0));
    let projection_jacobian = cameras[j].get_jacobian_with_respect_to_position(&transformed_point.fixed_rows::<3>(0));
    let row_idx = ((i/3)*state.n_cams+j)*2;
    jacobian.fixed_slice_mut::<2,6>(row_idx,j).copy_from(&(projection_jacobian*lie_jacobian));
}

pub fn compute_jacobian<C : Camera>(state: &State, cameras: &Vec<&C>, jacobian: &mut DMatrix<Float>) -> () {
    for j in (0..6*state.n_cams).step_by(6) {
        let u = state.data.fixed_rows::<3>(j);
        let w = state.data.fixed_rows::<3>(j+3);
        let transformation = exp(&u,&w);

        for i in ((6*state.n_cams)..state.data.ncols()).step_by(3) {
            compute_jacobian_wrt_camera_parameters(state, cameras , &transformation,i,j, jacobian);
            compute_jacobian_wrt_object_points(state, cameras, &transformation,i,j, jacobian);

        }
    }
}

pub fn optimize<C : Camera>(state: &mut State, cameras: &Vec<&C>, observed_features: &mut DVector<Float>, runtime_parameters: &RuntimeParameters ) -> () {

    
    let mut new_state = state.clone();
    let state_size = state.n_cams+state.n_points;
    let mut jacobian = DMatrix::<Float>::zeros(observed_features.nrows(),state_size);
    let mut rescaled_jacobian_target = DMatrix::<Float>::zeros(observed_features.nrows(),state_size);
    let identity = DMatrix::<Float>::identity(state_size, state_size);
    let mut residuals = DVector::<Float>::zeros(observed_features.nrows());
    let mut rescaled_residuals_target = DVector::<Float>::zeros(observed_features.nrows());
    let mut new_residuals = DVector::<Float>::zeros(observed_features.nrows());
    let mut estimated_features = DVector::<Float>::zeros(observed_features.nrows());
    let mut new_estimated_features = DVector::<Float>::zeros(observed_features.nrows());
    let mut weights_vec = DVector::<Float>::from_element(observed_features.nrows(),1.0);


    get_estimated_features(state, cameras, &mut estimated_features);
    compute_residual(&estimated_features, observed_features, &mut residuals);

    if runtime_parameters.weighting {
        norm(
            &residuals,
            &runtime_parameters.intensity_weighting_function,
            &mut weights_vec,
        );
    }
    
    compute_jacobian(&state,&cameras,&mut jacobian);

    weight_residuals_sparse(&mut residuals, &weights_vec);
    weight_jacobian_sparse_dynamic(&mut jacobian, &weights_vec);



    let mut max_norm_delta = float::MAX;
    let mut delta_thresh = float::MIN;
    let mut delta_norm = float::MAX;
    let mut nu = 2.0;

    let mut mu: Option<Float> = match runtime_parameters.lm {
        true => None,
        false => Some(0.0)
    };
    let step = match runtime_parameters.lm {
        true => 1.0,
        false => runtime_parameters.step_sizes[0]
    };
    let tau = runtime_parameters.taus[0];
    let max_iterations = runtime_parameters.max_iterations[0];
    
    let mut cost = compute_cost(&residuals,&runtime_parameters.loss_function);
    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (cost.sqrt() > runtime_parameters.eps[0])) || (runtime_parameters.lm && (delta_norm > delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))  && iteration_count < max_iterations  {
        if runtime_parameters.debug{
            println!("it: {}, avg_rmse: {}",iteration_count,cost.sqrt());
        }


        let (delta,g,gain_ratio_denom, mu_val) 
            = gauss_newton_step_with_loss_and_schur(&residuals,
                &jacobian,
                &identity,
                mu,
                tau,
                cost,
                &runtime_parameters.loss_function,
                &mut rescaled_jacobian_target,
                &mut rescaled_residuals_target); //TODO
        mu = Some(mu_val);


        let pertb = step*(&delta);
        new_state.update(&pertb);
        get_estimated_features(state, cameras, &mut new_estimated_features);
        compute_residual(&estimated_features, observed_features, &mut new_residuals);
        weight_residuals_sparse(&mut new_residuals, &weights_vec);



        let new_cost = compute_cost(&new_residuals,&runtime_parameters.loss_function);
        let cost_diff = cost-new_cost;
        let gain_ratio = cost_diff/gain_ratio_denom;
        if runtime_parameters.debug {
            println!("delta: {}, g: {}",delta,g);
            println!("cost: {}, new cost: {}",cost,new_cost);
        }

        if gain_ratio >= 0.0  || !runtime_parameters.lm {
            estimated_features.copy_from(&new_estimated_features);
            state.data.copy_from(&new_state.data); 

            cost = new_cost;

            max_norm_delta = max_norm_dynamic(&g);
            delta_norm = pertb.norm(); 

            delta_thresh = runtime_parameters.delta_eps*(estimated_features.norm() + runtime_parameters.delta_eps);

            residuals.copy_from(&new_residuals);

            

            compute_jacobian(&state,&cameras,&mut jacobian);
            weight_jacobian_sparse_dynamic(&mut jacobian, &weights_vec);

            let v: Float = 1.0/3.0;
            mu = Some(mu.unwrap() * v.max(1.0-(2.0*gain_ratio-1.0).powi(3)));
            nu = 2.0;
        } else {

            mu = Some(nu*mu.unwrap());
            nu *= 2.0;
        }

        iteration_count += 1;

    }

    
}


