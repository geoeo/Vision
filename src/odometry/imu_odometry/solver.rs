extern crate nalgebra as na;

use na::{Vector3,Matrix4,Isometry3,Rotation3,SMatrix,SVector, Const, DimMin};

use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::{imu_odometry, imu_odometry::imu_delta::ImuDelta,imu_odometry::bias, imu_odometry::bias::{BiasDelta,BiasPreintegrated}, imu_odometry::{ImuResidual, ImuCovariance}};
use crate::sensors::imu::imu_data_frame::ImuDataFrame;
use crate::numerics::{max_norm, least_squares::{weight_residuals,weight_jacobian, gauss_newton_step}};
use crate::{Float,float};

const OBSERVATIONS_DIM: usize = 9;
const PARAMETERS_DIM: usize = 15; //With bias


pub fn run_trajectory(imu_data_measurements: &Vec<ImuDataFrame>, gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters<Float>) -> Vec<Isometry3<Float>> {
    let mut bias_delta = BiasDelta::empty();
    imu_data_measurements.iter().enumerate().map(|(i,measurement)| {
        let (transform_est, bias_update) = run(i+1,measurement,&bias_delta,gravity_body,runtime_parameters);
        bias_delta = bias_delta.add_delta(&bias_update);
        let rotation = Rotation3::<Float>::from_matrix(&transform_est.fixed_view::<3,3>(0,0).into_owned());
        Isometry3::<Float>::new(transform_est.fixed_view::<3,1>(0,3).into_owned(),rotation.scaled_axis())
    }).collect::<Vec<Isometry3<Float>>>()
}

pub fn run<>(iteration: usize, imu_data_measurement: &ImuDataFrame, prev_bias_delta: &BiasDelta, gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters<Float>) -> (Matrix4<Float>, BiasDelta) {

    let imu_data_measurement_with_bias = imu_data_measurement.new_from_bias(prev_bias_delta);
    let (preintegrated_measurement, imu_covariance, preintegrated_bias) = imu_odometry::pre_integration(&imu_data_measurement_with_bias, gravity_body);
    let mut mat_result = Matrix4::<Float>::identity();
    
    let result = estimate::<OBSERVATIONS_DIM,PARAMETERS_DIM>(&imu_data_measurement_with_bias,&preintegrated_measurement,&preintegrated_bias, &imu_covariance ,&mat_result,gravity_body,runtime_parameters);
    mat_result = result.0;
    let solver_iterations = result.1;
    let bias_delta = result.2;


    if runtime_parameters.show_octave_result {
        println!("{}, est_transform: {}, solver iterations: {}",iteration,mat_result, solver_iterations);
    }

    

    if runtime_parameters.show_octave_result {
        println!("final: est_transform: {}",mat_result);
    }

    (mat_result, bias_delta)
}

//TODO: buffer all debug strings and print at the end. Also the numeric matricies could be buffered per octave level
fn estimate<const R: usize, const C: usize>(imu_data_measurement: &ImuDataFrame, preintegrated_measurement: &ImuDelta,preintegrated_bias: &BiasPreintegrated ,imu_covariance: &ImuCovariance,
    initial_guess_mat: &Matrix4<Float>,gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters<Float>) -> (Matrix4<Float>, usize, BiasDelta) where Const<C>: DimMin<Const<C>, Output = Const<C>> {
    let identity = SMatrix::<Float,C,C>::identity();
    let mut residuals_imu = SVector::<Float,R>::zeros();
    let mut jacobian_full = SMatrix::<Float,R,C>::zeros();

    let mut est_transform = initial_guess_mat.clone();
    let mut estimate = ImuDelta::empty();
    let mut bias_estimate = BiasDelta::empty();
    let delta_t = imu_data_measurement.gyro_ts[imu_data_measurement.gyro_ts.len()-1] - imu_data_measurement.gyro_ts[0]; // Gyro has a higher sampling rate

    let mut residuals = imu_odometry::generate_residual(&estimate, preintegrated_measurement,&bias_estimate,preintegrated_bias);
    //let mut residuals_unweighted = residuals.clone();

    let mut bias_a_residuals = bias::compute_residual(&bias_estimate.bias_a_delta, &preintegrated_bias.integrated_bias_a);
    let mut bias_g_residuals = bias::compute_residual(&bias_estimate.bias_g_delta, &preintegrated_bias.integrated_bias_g);

    let weights = match imu_covariance.cholesky() {
        Some(v) => v.inverse(),
        None => {
            println!("Warning Cholesky failed for imu covariance");
            ImuCovariance::identity()
        }
    };
    let weight_l_upper = weights.cholesky().expect("Cholesky Decomp Failed!").l().transpose();
    let mut jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);

    weight_residuals::<_,9>(&mut residuals, &weight_l_upper);
    weight_jacobian::<_,9,9>(&mut jacobian, &weight_l_upper);


    //let mut bias_jacobian = bias::genrate_residual_jacobian(&bias_estimate, preintegrated_bias, &residuals_unweighted);
    let mut bias_jacobian = bias::genrate_residual_jacobian(&bias_estimate, preintegrated_bias, &residuals);

    bias::weight_residual(&mut bias_a_residuals, &preintegrated_bias.bias_a_std);
    bias::weight_residual(&mut bias_g_residuals, &preintegrated_bias.bias_g_std);


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
    
    let mut cost = compute_cost(&residuals) + bias::compute_cost_for_weighted(&bias_a_residuals, &bias_g_residuals);
    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (cost.sqrt() > runtime_parameters.eps[0])) || (runtime_parameters.lm && (delta_norm > delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))  && iteration_count < max_iterations  {
        if runtime_parameters.debug{
            println!("it: {}, avg_rmse: {}",iteration_count,cost.sqrt());
        }

        residuals_imu.fixed_rows_mut::<9>(0).copy_from(&residuals);
        jacobian_full.fixed_view_mut::<9,9>(0,0).copy_from(&jacobian);
        jacobian_full.fixed_view_mut::<9,6>(0,9).copy_from(&bias_jacobian);

        let (delta,g,gain_ratio_denom, mu_val) 
            = gauss_newton_step(&residuals_imu, &jacobian_full, &identity, mu, tau);
        mu = Some(mu_val);


        let pertb = step*delta;
        let new_estimate = estimate.add_pertb(&pertb.fixed_rows::<9>(0));
        let new_bias_estimate = bias_estimate.add_pertb(&pertb.fixed_rows::<6>(9));

        let mut new_residuals = imu_odometry::generate_residual(&new_estimate, preintegrated_measurement, &new_bias_estimate, preintegrated_bias);
        //let new_residuals_unweighted = new_residuals.clone();
        weight_residuals::<_,9>(&mut new_residuals, &weight_l_upper);

        let mut new_bias_a_residuals = bias::compute_residual(&new_bias_estimate.bias_a_delta, &preintegrated_bias.integrated_bias_a);
        let mut new_bias_g_residuals = bias::compute_residual(&new_bias_estimate.bias_g_delta, &preintegrated_bias.integrated_bias_g);


        bias::weight_residual(&mut new_bias_a_residuals, &preintegrated_bias.bias_a_std);
        bias::weight_residual(&mut new_bias_g_residuals, &preintegrated_bias.bias_g_std);

        let new_cost = compute_cost(&new_residuals) + bias::compute_cost_for_weighted(&new_bias_a_residuals, &new_bias_g_residuals);
        let cost_diff = cost-new_cost;
        let gain_ratio = cost_diff/gain_ratio_denom;
        if runtime_parameters.debug {
            println!("delta: {}, g: {}",delta,g);
            println!("cost: {}, new cost: {}",cost,new_cost);
            println!("bias delta a {}, bias delta g: {}",new_bias_estimate.bias_a_delta,new_bias_estimate.bias_g_delta);
        }

        if gain_ratio >= 0.0  || !runtime_parameters.lm {
            estimate = new_estimate;
            bias_estimate = new_bias_estimate;
            est_transform = estimate.get_pose().to_matrix();

            cost = new_cost;

            max_norm_delta = max_norm(&g);
            delta_norm = pertb.norm(); 

            delta_thresh = runtime_parameters.delta_eps*(estimate.norm() + bias_estimate.norm() + runtime_parameters.delta_eps);

            residuals.copy_from(&new_residuals);
            //residuals_unweighted.copy_from(&new_residuals_unweighted);
            bias_a_residuals.copy_from(&new_bias_a_residuals);
            bias_g_residuals.copy_from(&new_bias_g_residuals);
            

            jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);
            weight_jacobian::<_, 9,9>(&mut jacobian, &weight_l_upper);

            bias_jacobian = bias::genrate_residual_jacobian(&bias_estimate, preintegrated_bias, &residuals);

            let v: Float = 1.0/3.0;
            mu = Some(mu.unwrap() * v.max(1.0-(2.0*gain_ratio-1.0).powi(3)));
            nu = 2.0;
        } else {

            mu = Some(nu*mu.unwrap());
            nu *= 2.0;
        }

        iteration_count += 1;

    }

    // println!{"bias delta a: {}", bias_estimate.bias_a_delta};
    // println!{"bias delta g: {}", bias_estimate.bias_g_delta};
    // println!{"bias a: {}", bias_estimate.bias_g_delta};

    (est_transform,iteration_count, bias_estimate)
}


fn compute_cost(residuals: &ImuResidual) -> Float {
    (residuals.transpose()*residuals)[0]
}

