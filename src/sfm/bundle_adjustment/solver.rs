extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use na::{DVector,DMatrix,Matrix, Dynamic, U4, VecStorage,Point3, Vector4, ComplexField, base::Scalar, RealField};
use num_traits::{float,cast};
use crate::sensors::camera::Camera;
use crate::numerics::lie::left_jacobian_around_identity;
use crate::numerics::{max_norm, least_squares::{compute_cost,weight_jacobian_sparse,weight_residuals_sparse, calc_weight_vec, gauss_newton_step_with_schur, gauss_newton_step_with_conguate_gradient}};
use crate::sfm::{landmark::Landmark,bundle_adjustment::{state::State, camera_feature_map::CameraFeatureMap}};
use crate::odometry::runtime_parameters::RuntimeParameters; //TODO remove dependency on odometry module

const CAMERA_PARAM_SIZE: usize = 6; //TODO make this generic with state

pub fn get_feature_index_in_residual(cam_id: usize, feature_id: usize, n_cams: usize) -> usize {
    2*(cam_id + feature_id*n_cams)
}

/**
 * In the format [f1_cam1, f1_cam2,...]
 * Some entries may be 0 since not all cams see all points
 * */
pub fn get_estimated_features<F, C : Camera<F>, L: Landmark<F,T> + Copy + Clone, const T: usize>(
    state: &State<F,L,T>, cameras: &Vec<&C>,observed_features: &DVector<F>, estimated_features: &mut DVector<F>) 
    -> () where F: float::Float + Scalar + ComplexField + RealField {
    let n_cams = state.n_cams;
    let n_points = state.n_points;
    assert_eq!(estimated_features.nrows(),2*n_points*n_cams);
    let mut position_world = Matrix::<F,U4,Dynamic, VecStorage<F,U4,Dynamic>>::from_element(n_points, F::one());
    for j in 0..n_points {
        position_world.fixed_slice_mut::<3,1>(0,j).copy_from(&state.get_landmarks()[j].get_euclidean_representation().coords); 
    };
    for i in 0..n_cams {
        let cam_idx = 6*i;
        let pose = state.to_se3(cam_idx);
        let camera = &cameras[i];

        //TODO: use transform_into_other_camera_frame
        let transformed_points = pose*&position_world;
        for j in 0..n_points {
            let estimated_feature = camera.project(&transformed_points.fixed_slice::<3,1>(0,j));  
            
            let feat_id = get_feature_index_in_residual(i, j, n_cams);
            // If at least one camera has no match, skip
            if !(observed_features[feat_id] == cast(CameraFeatureMap::NO_FEATURE_FLAG).unwrap() || observed_features[feat_id+1] == cast(CameraFeatureMap::NO_FEATURE_FLAG).unwrap()){
                estimated_features[feat_id] = estimated_feature.x;
                estimated_features[feat_id+1] = estimated_feature.y;
            }

        }

    }
}


pub fn compute_residual<F>(
    estimated_features: &DVector<F>, observed_features: &DVector<F>, residual_vector: &mut DVector<F>) 
    -> () where F: float::Float + Scalar{
    assert_eq!(residual_vector.nrows(), estimated_features.nrows());
    for i in 0..residual_vector.nrows() {
        if observed_features[i] != cast(CameraFeatureMap::NO_FEATURE_FLAG).unwrap() {
            residual_vector[i] =  estimated_features[i] - observed_features[i];
        } else {
            residual_vector[i] = F::zero();
        }
    }
}

pub fn compute_jacobian_wrt_object_points<F, C : Camera<F>, L: Landmark<F, T> + Copy + Clone, const T: usize>(camera: &C, state: &State<F,L,T>, cam_idx: usize, point_idx: usize, i: usize, j: usize, jacobian: &mut DMatrix<F>) 
    -> () where F: float::Float + Scalar + ComplexField + RealField {
    let transformation = state.to_se3(cam_idx);
    let point = state.get_landmarks()[point_idx].get_euclidean_representation();
    let jacobian_world = state.jacobian_wrt_world_coordiantes(point_idx,cam_idx);
    let transformed_point = transformation*Vector4::<F>::new(point[0],point[1],point[2],F::one());
    let projection_jacobian = camera.get_jacobian_with_respect_to_position_in_camera_frame(&transformed_point.fixed_rows::<3>(0));
    let local_jacobian = projection_jacobian*jacobian_world;

    jacobian.fixed_slice_mut::<2,T>(i,j).copy_from(&local_jacobian.fixed_slice::<2,T>(0,0));
}

pub fn compute_jacobian_wrt_camera_extrinsics<F, C : Camera<F>, L: Landmark<F, T> + Copy + Clone, const T: usize>(camera: &C, state: &State<F,L,T>, cam_idx: usize, point: &Point3<F> ,i: usize, j: usize, jacobian: &mut DMatrix<F>) 
    -> () where F: float::Float + Scalar + ComplexField + RealField {
    let transformation = state.to_se3(cam_idx);
    let transformed_point = transformation*Vector4::<F>::new(point[0],point[1],point[2],F::one());
    let lie_jacobian = left_jacobian_around_identity(&transformed_point.fixed_rows::<3>(0)); 

    let projection_jacobian = camera.get_jacobian_with_respect_to_position_in_camera_frame(&transformed_point.fixed_rows::<3>(0));
    let local_jacobian = projection_jacobian*lie_jacobian;

    jacobian.fixed_slice_mut::<2,6>(i,j).copy_from(&local_jacobian);
}

pub fn compute_jacobian<F, C : Camera<F>, L: Landmark<F, T> + Copy + Clone, const T: usize>(state: &State<F,L,T>, cameras: &Vec<&C>, jacobian: &mut DMatrix<F>) 
    -> ()  where F: float::Float + Scalar + ComplexField + RealField {
    //cam
    let number_of_cam_params = 6*state.n_cams;
    for cam_state_idx in (0..number_of_cam_params).step_by(6) {
        let cam_id = cam_state_idx/6;
        let camera = cameras[cam_id];
        
        //landmark
        for point_id in 0..state.n_points {
            let point = state.get_landmarks()[point_id].get_euclidean_representation();

            let row = get_feature_index_in_residual(cam_id, point_id, state.n_cams);
            let a_j = cam_state_idx;
            let b_j = number_of_cam_params+(T*point_id);
            

            compute_jacobian_wrt_camera_extrinsics(camera , state, cam_state_idx,&point,row,a_j, jacobian);
            compute_jacobian_wrt_object_points(camera, state, cam_state_idx ,point_id,row,b_j, jacobian);

        }

    }

}

pub fn optimize<F, C : Camera<F>, L: Landmark<F, LANDMARK_PARAM_SIZE> + Copy + Clone, const LANDMARK_PARAM_SIZE: usize>(state: &mut State<F,L,LANDMARK_PARAM_SIZE>, cameras: &Vec<&C>, observed_features: &DVector<F>, runtime_parameters: &RuntimeParameters<F> ) 
    -> Option<Vec<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; LANDMARK_PARAM_SIZE]>)>> where F: float::Float + Scalar + ComplexField + RealField{
    

    let max_iterations = runtime_parameters.max_iterations[0];

    let u_span = CAMERA_PARAM_SIZE*state.n_cams;
    let v_span = LANDMARK_PARAM_SIZE*state.n_points;
    
    let mut new_state = state.clone();
    let state_size = CAMERA_PARAM_SIZE*state.n_cams+LANDMARK_PARAM_SIZE*state.n_points;
    let mut jacobian = DMatrix::<F>::zeros(observed_features.nrows(),state_size); // a lot of memory
    let mut residuals = DVector::<F>::zeros(observed_features.nrows());
    let mut new_residuals = DVector::<F>::zeros(observed_features.nrows());
    let mut estimated_features = DVector::<F>::zeros(observed_features.nrows());
    let mut new_estimated_features = DVector::<F>::zeros(observed_features.nrows());
    let mut weights_vec = DVector::<F>::from_element(observed_features.nrows(),F::one());
    let mut target_arrowhead = DMatrix::<F>::zeros(state_size, state_size); // a lot of memory
    let mut g = DVector::<F>::from_element(state_size,F::zero()); 
    let mut delta = DVector::<F>::from_element(state_size,F::zero());
    let mut debug_state_list = match runtime_parameters.debug {
        true => Some(Vec::<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; LANDMARK_PARAM_SIZE]>)>::with_capacity(max_iterations)),
        false => None
    };
    let mut v_star_inv = DMatrix::<F>::zeros(v_span,v_span); // a lot of memory - maybe use sparse format
    let mut preconditioner = DMatrix::<F>::zeros(u_span,u_span); // a lot of memory - maybe use sparse format
    let two : F = num_traits::cast(2.0).unwrap();

    println!("BA Memory Allocation Complete.");

    get_estimated_features(state, cameras,observed_features, &mut estimated_features);
    compute_residual(&estimated_features, observed_features, &mut residuals);
    compute_jacobian(&state,&cameras,&mut jacobian);

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
    while ((!runtime_parameters.lm && (float::Float::sqrt(cost) > runtime_parameters.eps[0])) || 
    (runtime_parameters.lm && delta_norm > delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps && float::Float::sqrt(cost) > runtime_parameters.eps[0] ))  && iteration_count < max_iterations  {
        println!("it: {}, avg_rmse: {}",iteration_count,float::Float::sqrt(cost));
        if runtime_parameters.debug {
            debug_state_list.as_mut().expect("Debug is true but state list is None!. This should not happen").push(state.to_serial());
        }

        target_arrowhead.fill(F::zero());
        g.fill(F::zero());
        delta.fill(F::zero());
        v_star_inv.fill(F::zero());

        //TODO: switch in runtime parameters
    // let gauss_newton_result 
    //     = gauss_newton_step_with_schur::<_,_,_,_,_,_,_,LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>(
    //         &mut target_arrowhead,
    //         &mut g,
    //         &mut delta,
    //         &mut v_star_inv,
    //         &residuals,
    //         &jacobian,
    //         mu,
    //         tau,
    //         state.n_cams,
    //         state.n_points,
    //         u_span,
    //         v_span
    //     ); 


    preconditioner.fill(F::zero());
    let gauss_newton_result 
        = gauss_newton_step_with_conguate_gradient::<_,_,_,_,_,_,_,LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>(
            &mut target_arrowhead,
            &mut g,
            &mut delta,
            &mut v_star_inv,
            &mut preconditioner,
            &residuals,
            &jacobian,
            mu,
            tau,
            state.n_cams,
            state.n_points,
            u_span,
            v_span,
            runtime_parameters.cg_threshold,
            runtime_parameters.cg_max_it
           ); 

        let (gain_ratio, new_cost, pertb_norm, cost_diff) = match gauss_newton_result {
            Some((gain_ratio_denom, mu_val)) => {
                mu = Some(mu_val);
                let pertb = delta.scale(step);
                new_state.update(&pertb);
        
                get_estimated_features(&new_state, cameras,observed_features, &mut new_estimated_features);
                compute_residual(&new_estimated_features, observed_features, &mut new_residuals);
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
                (gain_ratio, new_cost, pertb.norm(), cost_diff)
            },
            None => (float::Float::nan(), float::Float::nan(), float::Float::nan(), float::Float::nan())
        };

        println!("cost: {}, new cost: {}, mu: {:?}, gain: {} , nu: {}, std: {:?}",cost,new_cost, mu, gain_ratio, nu, std);
        
        if (!gain_ratio.is_nan() && gain_ratio > F::zero() && cost_diff > F::zero()) || !runtime_parameters.lm {
            estimated_features.copy_from(&new_estimated_features);
            state.copy_from(&new_state); 

            cost = new_cost;

            max_norm_delta = max_norm(&g);
            delta_norm = pertb_norm; 

            delta_thresh = runtime_parameters.delta_eps*(estimated_features.norm() + runtime_parameters.delta_eps);

            residuals.copy_from(&new_residuals);

            jacobian.fill(F::zero());
            compute_jacobian(&state,&cameras,&mut jacobian);
            if std.is_some() {
                weight_jacobian_sparse(&mut jacobian, &weights_vec);
            }

            let v: F = cast(1.0 / 3.0).unwrap();
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

    }

    debug_state_list
}


