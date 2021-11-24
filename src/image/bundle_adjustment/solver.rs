extern crate nalgebra as na;

use na::{DVector,DMatrix,Matrix, Dynamic, U4, VecStorage,Vector, Vector4, Matrix2x4,Matrix2x3,Matrix1x6,Matrix1x3, Matrix4,U3,U1,base::storage::Storage};
use crate::{float,Float};
use crate::sensors::camera::Camera;
use crate::numerics::lie::{exp, left_jacobian_around_identity};
use crate::numerics::{max_norm, solver::{compute_cost,weight_jacobian_sparse,weight_residuals_sparse, calc_weight_vec, gauss_newton_step, gauss_newton_step_with_loss_and_schur, gauss_newton_step_with_schur}};
use crate::image::bundle_adjustment::state::State;
use crate::odometry::runtime_parameters::RuntimeParameters; //TODO remove dependency on odometry module

/**
 * In the format [f1_cam1_x,f1_cam1_y,f2_cam1_x,f2_cam1_y,f3_cam1_x,f3_cam1_y,f1_cam2_x,f1_cam2_y,f2_cam2_x,...]
 * Some entries may be 0 since not all cams see all points
 * */
pub fn get_estimated_features<C : Camera>(state: &State, cameras: &Vec<C>, estimated_features: &mut DVector<Float>) -> () {
    let n_cams = state.n_cams;
    let n_cam_parameters = 6*n_cams;
    let n_points = state.n_points;
    let estimated_state = &state.data;
    assert_eq!(estimated_features.nrows(),2*n_points*n_cams);
    for i in 0..n_cams {
        let cam_idx = 6*i*n_cams;
        let u = estimated_state.fixed_rows::<3>(cam_idx);
        let w = estimated_state.fixed_rows::<3>(cam_idx+3);
        let pose = exp(&u,&w);
        let camera = &cameras[i];
        let offset = 2*i*n_points;

        let mut position_per_cam = Matrix::<Float,U4,Dynamic, VecStorage<Float,U4,Dynamic>>::from_element(n_points, 1.0);
        for j in 0..n_points {
            position_per_cam.fixed_slice_mut::<3,1>(0,j).copy_from(&estimated_state.fixed_rows::<3>(n_cam_parameters+3*j)); 
        };
        let transformed_points = pose*position_per_cam;
        for j in 0..n_points {
            let estimated_feature = camera.project(&transformed_points.fixed_slice::<3,1>(0,j));            
            estimated_features[offset+2*j] = estimated_feature.x;
            estimated_features[offset+2*j+1] = estimated_feature.y;
        }

    }

}


pub fn compute_residual(estimated_features: &DVector<Float>, observed_features: &DVector<Float>, residual_vector: &mut DVector<Float>) -> () {
    assert_eq!(residual_vector.nrows(), estimated_features.nrows());
    for i in 0..residual_vector.nrows() {
        residual_vector[i] =  estimated_features[i] - observed_features[i];
    }
}


pub fn compute_jacobian_wrt_object_points<C : Camera,T>(camera: &C, transformation: &Matrix4<Float>, point: &Vector<Float,U3,T>, i: usize, j: usize, jacobian: &mut DMatrix<Float>) 
    -> () where T: Storage<Float,U3,U1> {
    let homogeneous_point = transformation*Vector4::<Float>::new(point[0],point[1],point[2],1.0);
    let mut homogeneous_projection_jacobian = Matrix2x4::<Float>::zeros();
    homogeneous_projection_jacobian.fixed_slice_mut::<2,3>(0,0).copy_from(&camera.get_jacobian_with_respect_to_position(&homogeneous_point.fixed_slice::<3,1>(0,0)));

    let local_jacobian = homogeneous_projection_jacobian*transformation;
    jacobian.fixed_slice_mut::<2,3>(i,j).copy_from(&local_jacobian.fixed_slice::<2,3>(0,0));
}

pub fn compute_jacobian_wrt_camera_parameters<C : Camera, T>( camera: &C, transformation: &Matrix4<Float>, point: &Vector<Float,U3,T> ,i: usize, j: usize, jacobian: &mut DMatrix<Float>) 
    -> () where T: Storage<Float,U3,U1> {
    let transformed_point = transformation*Vector4::<Float>::new(point[0],point[1],point[2],1.0);
    let lie_jacobian = left_jacobian_around_identity(&transformed_point.fixed_rows::<3>(0));

    let mut projection = Matrix2x3::<Float>::zeros();
    projection.fixed_slice_mut::<2,3>(0,0).copy_from(&camera.get_projection().fixed_slice::<2,3>(0,0));

    let mut homogeneous_projection_jacobian = Matrix2x3::<Float>::zeros();
    homogeneous_projection_jacobian.fixed_slice_mut::<2,3>(0,0).copy_from(&camera.get_jacobian_with_respect_to_position(&transformed_point.fixed_slice::<3,1>(0,0)));

    jacobian.fixed_slice_mut::<2,6>(i,j).copy_from(&(homogeneous_projection_jacobian*lie_jacobian));
}

pub fn compute_jacobian<C : Camera>(state: &State, cameras: &Vec<C>, jacobian: &mut DMatrix<Float>) -> () {
    //cam
    let number_of_cam_params = 6*state.n_cams;
    for cam_state_idx in (0..number_of_cam_params).step_by(6) {
        let cam_id = cam_state_idx/6;
        let camera = &cameras[cam_id];

        let u = state.data.fixed_rows::<3>(cam_state_idx);
        let w = state.data.fixed_rows::<3>(cam_state_idx+3);
        let transformation = exp(&u,&w);
        
        //point
        for point_state_idx in (number_of_cam_params..state.data.nrows()).step_by(3) {
            let point = &state.data.fixed_rows::<3>(point_state_idx);

            let point_id = (point_state_idx-number_of_cam_params)/3;
            let a_i = 2*(cam_id+point_id*state.n_cams);
            let a_j = cam_state_idx;
            let b_i = 2*(state.n_cams*point_id+cam_id);
            let b_j = number_of_cam_params+point_id*3;

            compute_jacobian_wrt_camera_parameters(camera , &transformation,point,a_i,a_j, jacobian);
            compute_jacobian_wrt_object_points(camera, &transformation,point,b_i,b_j, jacobian);

        }

    }


}

pub fn optimize<C : Camera>(state: &mut State, cameras: &Vec<C>, observed_features: &DVector<Float>, runtime_parameters: &RuntimeParameters ) -> () {
    let mut new_state = state.clone();
    let state_size = 6*state.n_cams+3*state.n_points;
    let mut jacobian = DMatrix::<Float>::zeros(observed_features.nrows(),state_size);
    let mut residuals = DVector::<Float>::zeros(observed_features.nrows());
    let mut new_residuals = DVector::<Float>::zeros(observed_features.nrows());
    let mut estimated_features = DVector::<Float>::zeros(observed_features.nrows());
    let mut new_estimated_features = DVector::<Float>::zeros(observed_features.nrows());
    let mut weights_vec = DVector::<Float>::from_element(observed_features.nrows(),1.0);
    let mut target_arrowhead = DMatrix::<Float>::zeros(state_size, state_size);
    let mut g = DVector::<Float>::from_element(state_size,0.0); 
    let mut delta = DVector::<Float>::from_element(state_size,0.0); 

    let identity = DMatrix::<Float>::identity(state_size, state_size);


    get_estimated_features(state, cameras, &mut estimated_features);
    compute_residual(&estimated_features, observed_features, &mut residuals);

    compute_jacobian(&state,&cameras,&mut jacobian);

    weight_residuals_sparse(&mut residuals, &weights_vec); 
    weight_jacobian_sparse(&mut jacobian, &weights_vec);


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

        target_arrowhead.fill(0.0);
        g.fill(0.0);
        delta.fill(0.0);

        let (gain_ratio_denom, mu_val) 
            = gauss_newton_step_with_schur(
                &mut target_arrowhead,
                &mut g,
                &mut delta,
                &residuals,
                &jacobian,
                mu,
                tau,
                state.n_cams,
                state.n_points
            ); 

        // let (delta,g,gain_ratio_denom, mu_val) 
        //     = gauss_newton_step(&residuals,
        //          &jacobian,
        //          &identity,
        //          mu,
        //          tau); 



        mu = Some(mu_val);

        let pertb = step*(&delta);
        new_state.update(&pertb);

        get_estimated_features(&new_state, cameras, &mut new_estimated_features);
        compute_residual(&new_estimated_features, observed_features, &mut new_residuals);
        if runtime_parameters.weighting {
            calc_weight_vec(
                &new_residuals,
                &runtime_parameters.intensity_weighting_function,
                &mut weights_vec,
            );
        }
        weight_residuals_sparse(&mut new_residuals, &weights_vec);

        let new_cost = compute_cost(&new_residuals,&runtime_parameters.loss_function);
        let cost_diff = cost-new_cost;
        let gain_ratio = cost_diff/gain_ratio_denom;
        if runtime_parameters.debug {
            //println!("delta: {}, g: {}",delta,g);
            println!("cost: {}, new cost: {}, mu: {:?}, gain: {} , nu: {}",cost,new_cost, mu, gain_ratio, nu);
        }

        //TODO: check gain ratio calc
        if gain_ratio >= 0.0  || !runtime_parameters.lm {
            estimated_features.copy_from(&new_estimated_features);
            state.data.copy_from(&new_state.data); 

            cost = new_cost;

            max_norm_delta = max_norm(&g);
            delta_norm = pertb.norm(); 

            delta_thresh = runtime_parameters.delta_eps*(estimated_features.norm() + runtime_parameters.delta_eps);

            residuals.copy_from(&new_residuals);

            

            compute_jacobian(&state,&cameras,&mut jacobian);
            weight_jacobian_sparse(&mut jacobian, &weights_vec);

            let v: Float = 1.0/3.0;
            mu = Some(mu.unwrap() * v.max(1.0-(2.0*gain_ratio-1.0).powi(3)));
            nu = 2.0;
        } else {

            mu = Some(nu*mu.unwrap());
            nu *= 2.0;
        }

        iteration_count += 1;

        if mu.unwrap().is_infinite(){
            break;
        }

    }

    
}


