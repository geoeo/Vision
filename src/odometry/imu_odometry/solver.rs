extern crate nalgebra as na;

use na::{Vector3,Matrix4,SMatrix,SVector, Const, DimMin};

use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::{imu_odometry, imu_odometry::imu_delta::ImuDelta, imu_odometry::{ImuResidual,ImuJacobian,ImuPertrubation, ImuCovariance, weight_jacobian, weight_residuals}};
use crate::sensors::imu::imu_data_frame::ImuDataFrame;
use crate::numerics::max_norm;
use crate::{Float,float};

const OBSERVATIONS_DIM: usize = 9;
const PARAMETERS_DIM: usize = 15; //With bias


pub fn run_trajectory(imu_data_measurements: &Vec<ImuDataFrame>, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> Vec<Matrix4<Float>> {
    imu_data_measurements.iter().enumerate().map(|(i,measurement)| run(i+1,&measurement,bias_gyroscope,bias_accelerometer,gravity_body,runtime_parameters)).collect::<Vec<Matrix4<Float>>>()
}

pub fn run<>(iteration: usize, imu_data_measurement: &ImuDataFrame,bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, 
    gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> Matrix4<Float> {

    let (preintegrated_measurement, imu_covariance) = imu_odometry::pre_integration(imu_data_measurement, bias_gyroscope, bias_accelerometer, gravity_body);
    let mut mat_result = Matrix4::<Float>::identity();
    
    let result = estimate::<OBSERVATIONS_DIM,PARAMETERS_DIM>(imu_data_measurement,&preintegrated_measurement, &imu_covariance ,&mat_result,gravity_body,runtime_parameters);
    mat_result = result.0;
    let solver_iterations = result.1;

    if runtime_parameters.show_octave_result {
        println!("{}, est_transform: {}, solver iterations: {}",iteration,mat_result, solver_iterations);
    }

    

    if runtime_parameters.show_octave_result {
        println!("final: est_transform: {}",mat_result);
    }

    mat_result
}

//TODO: buffer all debug strings and print at the end. Also the numeric matricies could be buffered per octave level
fn estimate<const R: usize, const C: usize>(imu_data_measurement: &ImuDataFrame, preintegrated_measurement: &ImuDelta,imu_covariance: &ImuCovariance,
    initial_guess_mat: &Matrix4<Float>,gravity_body: &Vector3<Float>, runtime_parameters: &RuntimeParameters) -> (Matrix4<Float>, usize) where Const<C>: DimMin<Const<C>, Output = Const<C>> {
    let identity = SMatrix::<Float,C,C>::identity();
    let mut residuals_full = SVector::<Float,R>::zeros();
    let mut jacobian_full = SMatrix::<Float,R,C>::zeros();

    let mut est_transform = initial_guess_mat.clone();
    let mut estimate = ImuDelta::empty();
    let delta_t = imu_data_measurement.gyro_ts[imu_data_measurement.gyro_ts.len()-1] - imu_data_measurement.gyro_ts[0]; // Gyro has a higher sampling rate

    let mut residuals = imu_odometry::generate_residual(&estimate, preintegrated_measurement);
    let mut residuals_unweighted = residuals.clone();

    let weights = match imu_covariance.cholesky() {
        Some(v) => v.inverse(),
        None => {
            println!("Warning Cholesky failed for imu covariance");
            ImuCovariance::identity()
        }
    };
    let weight_l_upper = weights.cholesky().expect("Cholesky Decomp Failed!").l().transpose();
    let mut jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);

    weight_residuals(&mut residuals, &weight_l_upper);
    weight_jacobian(&mut jacobian, &weight_l_upper);


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
    
    let mut cost = compute_cost(&residuals);
    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (cost.sqrt() > runtime_parameters.eps[0])) || (runtime_parameters.lm && (delta_norm > delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))  && iteration_count < max_iterations  {
        if runtime_parameters.debug{
            println!("it: {}, avg_rmse: {}",iteration_count,cost.sqrt());
        }

        residuals_full.fixed_rows_mut::<9>(0).copy_from(&residuals);
        jacobian_full.fixed_slice_mut::<9,9>(0,0).copy_from(&jacobian);

        let (delta,g,gain_ratio_denom, mu_val) 
            = gauss_newton_step_with_loss(&residuals_full, &jacobian_full, &identity, mu, tau);
        mu = Some(mu_val);


        let pertb = step*delta;
        let new_estimate = estimate.add_pertb(&pertb.fixed_rows::<9>(0));
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

            cost = new_cost;

            max_norm_delta = max_norm(&g);
            delta_norm = pertb.norm(); 

            delta_thresh = runtime_parameters.delta_eps*(estimate.norm() + runtime_parameters.delta_eps);

            residuals.copy_from(&new_residuals);
            residuals_unweighted.copy_from(&new_residuals_unweighted);

            jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);
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

    (est_transform,iteration_count)
}


//TODO: potential for optimization. Maybe use less memory/matrices. 
#[allow(non_snake_case)]
fn gauss_newton_step_with_loss<const R: usize, const C: usize>(
    residuals: &SVector<Float, R>, 
    jacobian: &SMatrix<Float,R,C>,
    identity: &SMatrix<Float,C,C>,
     mu: Option<Float>, 
     tau: Float)-> (SVector<Float,C>,SVector<Float,C>,Float,Float) where Const<C>: DimMin<Const<C>, Output = Const<C>> {

    let (A,g) = (jacobian.transpose()*jacobian,jacobian.transpose()*residuals);
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

