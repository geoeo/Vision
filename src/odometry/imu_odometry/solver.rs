extern crate nalgebra as na;

use na::{U1,U2,U3,U4,U6,U9,RowVector2,Vector3,Vector4,Vector6,DVector,Matrix4,Matrix,Matrix6,MatrixN,DMatrix,Dynamic,VecStorage};
use std::boxed::Box;

use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::{imu_odometry, imu_odometry::imu_delta::ImuDelta, imu_odometry::{ImuResidual,ImuJacobian,ImuPertrubation, ImuCovariance}};
use crate::numerics::{lie,loss::LossFunction};
use crate::sensors::imu::imu_data_frame::ImuDataFrame;
use crate::{Float,float};


pub fn run_trajectory(imu_data_measurements: &Vec<ImuDataFrame>, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> Vec<Matrix4<Float>> {
    imu_data_measurements.iter().enumerate().map(|(i,measurement)| run(i+1,&measurement,bias_gyroscope,bias_accelerometer,gravity_body,runtime_parameters)).collect::<Vec<Matrix4<Float>>>()
}

pub fn run(iteration: usize, imu_data_measurement: &ImuDataFrame,bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> Matrix4<Float> {

    let (preintegrated_measurement, imu_covariance) = imu_odometry::pre_integration(imu_data_measurement, bias_gyroscope, bias_accelerometer, gravity_body);
    let mut lie_result = Vector6::<Float>::zeros();
    let mut mat_result = Matrix4::<Float>::identity();
    
    let result = estimate(imu_data_measurement,&preintegrated_measurement, &imu_covariance ,&lie_result,&mat_result,gravity_body,runtime_parameters);
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
fn estimate(imu_data_measurement: &ImuDataFrame, preintegrated_measurement: &ImuDelta,imu_covariance: &ImuCovariance ,initial_guess_lie: &Vector6<Float>,initial_guess_mat: &Matrix4<Float>,gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> (Vector6<Float>,Matrix4<Float>, usize) {
    let identity_9 = MatrixN::<Float,U9>::identity();

    let mut est_transform = initial_guess_mat.clone();
    let mut est_lie = initial_guess_lie.clone();
    let mut estimate = ImuDelta::empty();
    let delta_t = imu_data_measurement.gyro_ts[imu_data_measurement.gyro_ts.len()-1] - imu_data_measurement.gyro_ts[0]; // Gyro has a higher sampling rate

    let mut residuals = imu_odometry::generate_residual(&estimate, preintegrated_measurement);
    let mut residuals_unweighted = residuals.clone();

    // println!("{}", imu_covariance);

    let weights = match imu_covariance.cholesky() {
        Some(v) => v.inverse(),
        None => {
            println!("Warning Cholesky failed for imu covariance");
            identity_9
        }
    };
    let weight_l_upper = weights.cholesky().expect("Cholesky Decomp Failed!").l().transpose();
    let mut jacobian = imu_odometry::generate_jacobian(&est_lie.fixed_rows::<U3>(3), delta_t);
    let mut rescaled_jacobian_target = ImuJacobian::zeros(); 
    let mut rescaled_residual_target = ImuResidual::zeros();

    // println!("{}", jacobian);


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
    


    weight_residuals(&mut residuals, &weight_l_upper);
    weight_jacobian(&mut jacobian, &weight_l_upper);
    
    let mut cost = compute_cost(&residuals);
    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (cost.sqrt() > runtime_parameters.eps[0])) || (runtime_parameters.lm && (delta_norm >= delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))  && iteration_count < max_iterations  {
        if runtime_parameters.debug{
            println!("it: {}, avg_rmse: {}",iteration_count,cost.sqrt());
        }

        let (delta,g,gain_ratio_denom, mu_val) 
            = gauss_newton_step_with_loss(&residuals, &residuals_unweighted, &jacobian, &identity_9, mu, tau, cost, & runtime_parameters.loss_function, &mut rescaled_jacobian_target,&mut rescaled_residual_target);
        mu = Some(mu_val);


        let pertb = step*delta;
        let new_estimate = estimate.add_pertb(&pertb);
        let mut new_residuals = imu_odometry::generate_residual(&new_estimate, preintegrated_measurement);
        let new_residuals_unweighted = new_residuals.clone();

        weight_residuals(&mut new_residuals, &weight_l_upper);

        let new_cost = compute_cost(&new_residuals);
        let cost_diff = cost-new_cost;
        let gain_ratio = cost_diff/gain_ratio_denom;
        if runtime_parameters.debug {
            println!("delta: {}, g: {}",delta,g);
            println!("cost: {}, new cost: {}",cost,new_cost);
        }
        if gain_ratio >= 0.0  || !runtime_parameters.lm {
            estimate = new_estimate;
            est_transform = estimate.get_pose();
            est_lie = lie::ln(&est_transform);

            cost = new_cost;

            max_norm_delta = g.max();
            delta_norm = delta.norm();
            delta_thresh = runtime_parameters.delta_eps*(estimate.norm() + runtime_parameters.delta_eps);

            residuals = new_residuals.clone();
            residuals_unweighted = new_residuals_unweighted.clone();

            jacobian = imu_odometry::generate_jacobian(&est_lie.fixed_rows::<U3>(3), delta_t);
            weight_jacobian(&mut jacobian, &weight_l_upper);

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


fn weight_residuals(residual: &mut ImuResidual, weights: &MatrixN<Float,U9>) -> () {
    weights.mul_to(&residual.clone(),residual);
}

fn weight_jacobian(jacobian: &mut ImuJacobian, weights: &MatrixN<Float,U9>) -> () {
    weights.mul_to(&jacobian.clone(),jacobian);
}

//TODO: potential for optimization. Maybe use less memory/matrices. 
#[allow(non_snake_case)]
fn gauss_newton_step_with_loss(
    residuals: &ImuResidual, 
    residuals_unweighted: &ImuResidual, 
    jacobian: &ImuJacobian,
    identity: &MatrixN<Float,U9>,
     mu: Option<Float>, 
     tau: Float, 
     current_cost: Float, 
     loss_function: &Box<dyn LossFunction>,
      rescaled_jacobian_target: &mut ImuJacobian, 
      rescaled_residuals_target: &mut ImuResidual) -> (ImuPertrubation,ImuPertrubation,Float,Float) {

    let selected_root = loss_function.select_root(current_cost);
    let first_deriv_at_cost = loss_function.first_derivative_at_current(current_cost);
    let second_deriv_at_cost = loss_function.second_derivative_at_current(current_cost);
    let is_curvature_negative = second_deriv_at_cost*current_cost < -0.5*first_deriv_at_cost;
    let (A,g) = match selected_root { //TODO: check root selection

        root if root != 0.0 => {
            match is_curvature_negative {
                false => {
                    let first_derivative_sqrt = first_deriv_at_cost.sqrt();
                    let jacobian_factor = selected_root/current_cost;
                    let residual_scale = first_derivative_sqrt/(1.0-selected_root);
                    let res_j = residuals.transpose()*jacobian;
                    for i in 0..jacobian.nrows(){
                        rescaled_jacobian_target.row_mut(i).copy_from(&(first_derivative_sqrt*(jacobian.row(i) - (jacobian_factor*residuals[i]*res_j))));
                        rescaled_residuals_target[i] = residual_scale*residuals[i];
                    }
                    (rescaled_jacobian_target.transpose()*(rescaled_jacobian_target as &ImuJacobian),rescaled_jacobian_target.transpose()*(rescaled_residuals_target as &ImuResidual))
                }
                _ => (jacobian.transpose()*(first_deriv_at_cost*identity + 2.0*second_deriv_at_cost*residuals_unweighted*residuals_unweighted.transpose())*jacobian,first_deriv_at_cost*jacobian.transpose()*residuals) 
            }

        },
        _ => (jacobian.transpose()*jacobian,jacobian.transpose()*residuals) 
    };
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let decomp = (A+ mu_val*identity).qr();
    let h = decomp.solve(&(-g)).expect("QR Solve Failed");
    let gain_ratio_denom = h.transpose()*(mu_val*h-g);
    (h,g,gain_ratio_denom[0], mu_val)
}


fn compute_cost(residuals: &ImuResidual) -> Float {
    (residuals.transpose()*residuals)[0]
}

