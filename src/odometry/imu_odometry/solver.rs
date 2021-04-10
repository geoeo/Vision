extern crate nalgebra as na;

use na::{U1,U2,U3,U4,U6,RowVector2,Vector3,Vector4,Vector6,DVector,Matrix4,Matrix,Matrix6,MatrixN,DMatrix,Dynamic,VecStorage};
use std::boxed::Box;

use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::{imu_odometry, imu_odometry::imu_delta::ImuDelta, imu_odometry::ImuResidual};
use crate::numerics::{lie,loss::LossFunction};
use crate::sensors::imu::imu_data_frame::ImuDataFrame;
use crate::{Float,float};


pub fn run_trajectory(imu_data_measurements: &Vec<ImuDataFrame>, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> Vec<Matrix4<Float>> {
    imu_data_measurements.iter().enumerate().map(|(i,measurement)| run(i+1,&measurement,bias_gyroscope,bias_accelerometer,gravity_body,runtime_parameters)).collect::<Vec<Matrix4<Float>>>()
}

pub fn run(iteration: usize, imu_data_measurement: &ImuDataFrame,bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> Matrix4<Float> {

    let preintegrated_measurement = imu_odometry::pre_integration(imu_data_measurement, bias_gyroscope, bias_accelerometer, gravity_body);
    let mut lie_result = Vector6::<Float>::zeros();
    let mut mat_result = Matrix4::<Float>::identity();
    
    let result = estimate(imu_data_measurement,&preintegrated_measurement ,&lie_result,&mat_result,runtime_parameters,gravity_body);
    lie_result = result.0;
    mat_result = result.1;
    let solver_iterations = result.2;

    if runtime_parameters.show_octave_result {
        println!("{}, est_transform: {}, solver iterations: {}",iteration,mat_result, solver_iterations);
    }

    

    if runtime_parameters.show_octave_result {
        println!("final: est_transform: {}",mat_result);
    }

    mat_result
}

//TODO: buffer all debug strings and print at the end. Also the numeric matricies could be buffered per octave level
fn estimate(imu_data_measurement: &ImuDataFrame, preintegrated_measurement: &ImuDelta,initial_guess_lie: &Vector6<Float>,initial_guess_mat: &Matrix4<Float>,gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> (Vector6<Float>,Matrix4<Float>, usize) {
    let mut percentage_of_valid_pixels = 100.0;

    let identity_6 = Matrix6::<Float>::identity();

    let mut est_transform = initial_guess_mat.clone();
    let mut est_lie = initial_guess_lie.clone();
    let mut estimate = ImuDelta::empty();
    let mut imu_covariance = imu_odometry::ImuCovariance::zeros();
    let mut noise_covariance = &imu_data_measurement.noise_covariance;
    let delta_t = imu_data_measurement.gyro_ts[imu_data_measurement.gyro_ts.len()-1] - imu_data_measurement.gyro_ts[0]; // Gyro has a higher sampling rate

    let residuals = imu_odometry::generate_residual(&estimate, preintegrated_measurement);
    imu_covariance = imu_odometry::propagate_state_covariance(&imu_covariance, noise_covariance, imu_data_measurement, &preintegrated_measurement.delta_rotation_i_k, &preintegrated_measurement.delta_rotation_k, gravity_body);
    let weights = imu_covariance.cholesky().expect("Cholesky Decomp Failed!").inverse(); // Else try "try_inverse" or lu

    weight_residuals(&mut residuals, &weights);
    let mut cost = compute_cost(&residuals);

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
    
    let jacobian = imu_odometry::generate_jacobian(&est_lie.fixed_rows::<U3>(3), delta_t);


    weight_jacobian(&mut full_jacobian_weighted,&full_jacobian, &weights_vec);
    

    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (avg_cost.sqrt() > runtime_parameters.eps[octave_index])) || (runtime_parameters.lm && (delta_norm >= delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))  && iteration_count < max_iterations  {
        if runtime_parameters.debug{
            println!("it: {}, avg_rmse: {}, valid pixels: {}%",iteration_count,avg_cost.sqrt(),percentage_of_valid_pixels);
        }

        let (delta,g,gain_ratio_denom, mu_val) 
            = gauss_newton_step_with_loss(&residuals, &full_jacobian_weighted, &identity_6, mu, tau, cost, & runtime_parameters.loss_function, &mut rescaled_jacobian_target,&mut rescaled_residual_target);
        mu = Some(mu_val);


        //let new_est_lie = est_lie+ step*delta;
        //let new_est_transform = lie::exp(&new_est_lie.fixed_rows::<U3>(0),&new_est_lie.fixed_rows::<U3>(3));

        let pertb = step*delta;
        let new_est_transform = lie::exp(&pertb.fixed_slice::<U3, U1>(0, 0),&pertb.fixed_slice::<U3, U1>(3, 0))*est_transform;


        new_image_gradient_points.clear();
        compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&backprojected_points_flags,&new_est_transform, &intensity_camera , &mut new_residuals,&mut new_image_gradient_points);
        
        
        if runtime_parameters.weighting {
            compute_t_dist_weights(&new_residuals,&mut weights_vec,new_image_gradient_points.len() as Float,5.0,20,1e-10);
        }
        weight_residuals(&mut new_residuals, &weights_vec);

        percentage_of_valid_pixels = (new_image_gradient_points.len() as Float/number_of_pixels_float) *100.0;
        let new_cost = compute_cost(&new_residuals);
        let cost_diff = cost-new_cost;
        let gain_ratio = cost_diff/gain_ratio_denom;
        if runtime_parameters.debug {
            println!("{},{}",cost,new_cost);
        }
        if gain_ratio >= 0.0  || !runtime_parameters.lm {
            //est_lie = new_est_lie.clone();
            est_lie = lie::ln(&new_est_transform);
            est_transform = new_est_transform.clone();
            cost = new_cost;
            avg_cost = cost/new_image_gradient_points.len() as Float;

            max_norm_delta = g.max();
            delta_norm = delta.norm();
            delta_thresh = runtime_parameters.delta_eps*(est_lie.norm() + runtime_parameters.delta_eps);

            image_gradient_points = new_image_gradient_points.clone();
            residuals = new_residuals.clone();

            compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);
            compute_full_jacobian(&image_gradients,&constant_jacobians,&mut full_jacobian);
            weight_jacobian(&mut full_jacobian_weighted, &full_jacobian, &weights_vec);

            let v: Float = 1.0/3.0;
            mu = Some(mu.unwrap() * v.max(1.0-(2.0*gain_ratio-1.0).powi(3)));
            nu = 2.0;
        } else {

            mu = Some(nu*mu.unwrap());
            nu *= 2.0;
        }

        iteration_count += 1;

    }

    (est_lie,est_transform,iteration_count)
}



//No longer only diagonal of imu
fn weight_residuals(residual: &mut ImuResidual, weights: &MatrixN<Float,U9>) -> () {
    residual = weights*residual
}

//No longer only diagonal of imu
fn weight_jacobian(jacobian_target: &mut Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>, weights_vec: &DVector<Float>) -> () {
    panic!("not implemented")
}


#[allow(non_snake_case)]
fn gauss_newton_step(residuals_weighted: &DVector<Float>, jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,identity_6: &Matrix6<Float>, mu: Option<Float>, tau: Float) -> (Vector6<Float>,Vector6<Float>,Float,Float) {
    let A = jacobian.transpose()*jacobian;
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let g = jacobian.transpose()*residuals_weighted;
    let decomp = (A+ mu_val*identity_6).lu();
    let h = decomp.solve(&(-g)).expect("Linear resolution failed.");
    let gain_ratio_denom = h.transpose()*(mu_val*h-g);
    (h,g,gain_ratio_denom[0], mu_val)
}

//TODO: potential for optimization. Maybe use less memory/matrices. 
#[allow(non_snake_case)]
fn gauss_newton_step_with_loss(residuals: &DVector<Float>, 
    jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,
    identity_6: &Matrix6<Float>,
     mu: Option<Float>, 
     tau: Float, 
     current_cost: Float, 
     loss_function: &Box<dyn LossFunction>,
      rescaled_jacobian_target: &mut Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>, 
      rescaled_residuals_target: &mut DVector<Float>) -> (Vector6<Float>,Vector6<Float>,Float,Float) {

    let selected_root = loss_function.select_root(current_cost);
    let (A,g) =  match selected_root {

        root if root != 0.0 => {
            let first_derivative_sqrt = loss_function.first_derivative_at_current(current_cost).sqrt();
            let jacobian_factor = selected_root/current_cost;
            let residual_scale = first_derivative_sqrt/(1.0-selected_root);
            let res_j = residuals.transpose()*jacobian;
            for i in 0..jacobian.nrows(){
                rescaled_jacobian_target.row_mut(i).copy_from(&(first_derivative_sqrt*(jacobian.row(i) - (jacobian_factor*residuals[i]*res_j))));
                rescaled_residuals_target[i] = residual_scale*residuals[i];
            }
            (rescaled_jacobian_target.transpose()*rescaled_jacobian_target as &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,rescaled_jacobian_target.transpose()*rescaled_residuals_target as &DVector<Float>)
        },
        _ => (jacobian.transpose()*jacobian,jacobian.transpose()*residuals)
    };
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let decomp = (A+ mu_val*identity_6).lu();
    let h = decomp.solve(&(-g)).expect("Linear resolution failed.");
    let gain_ratio_denom = h.transpose()*(mu_val*h-g);
    (h,g,gain_ratio_denom[0], mu_val)
}


fn compute_cost(residuals: &ImuResidual) -> Float {
    (residuals.transpose()*residuals)[0]
}

