use nalgebra as na;

use na::{U3,U6,U9,Matrix3,Matrix6,MatrixN,MatrixMN,Vector3};
use crate::Float;
use crate::sensors::{DataFrame, imu::imu_data_frame::ImuDataFrame};
use crate::odometry::imu_odometry::imu_delta::ImuDelta;
use crate::numerics::lie::{exp_r,skew_symmetric,right_jacobian};

pub mod imu_delta;

pub type ImuCovariance = MatrixN<Float,U9>;
pub type NoiseCovariance = Matrix6<Float>;


//TODO: check gravity in formulae
#[allow(non_snake_case)]
pub fn pre_integration(imu_data: &ImuDataFrame, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>) -> ImuDelta {

    let accel_initial_time = imu_data.acceleration_ts[0];
    let accel_delta_times = imu_data.acceleration_ts[1..].iter().map(|t| t - accel_initial_time).collect::<Vec<Float>>();

    let gyro_initial_time = imu_data.gyro_ts[0];
    let gyro_delta_times = imu_data.gyro_ts[1..].iter().map(|t| t - gyro_initial_time).collect::<Vec<Float>>();

    let delta_rotations = imu_data.gyro_data[1..].iter().zip(gyro_delta_times.iter()).map(|(x,&dt)| (x-bias_gyroscope)*dt).map(|x| exp_r(&x)).collect::<Vec<Matrix3::<Float>>>();
    let delta_velocities = imu_data.acceleration_data[1..].iter().zip(delta_rotations.iter()).zip(accel_delta_times.iter()).map(|((x,dR),&dt)| dR*(x - bias_accelerometer + gravity_body)*dt).collect::<Vec<Vector3<Float>>>(); 
    let accumulated_velocities = delta_velocities.iter().scan(Vector3::<Float>::zeros(),|acc,dv| {
        *acc=*acc+*dv;
        Some(*acc)
    }).collect::<Vec<Vector3<Float>>>();
    let delta_positions = delta_velocities.iter().zip(accel_delta_times.iter()).zip(accumulated_velocities.iter()).map(|((dv,&dt),v_initial)| v_initial*dt +0.5*dv*dt).collect::<Vec<Vector3::<Float>>>(); 

    let identity = Matrix3::<Float>::identity();
    let empty_vector = Vector3::<Float>::zeros();

    let delta_rotation = delta_rotations.iter().fold(identity,|acc,x| acc*x);
    let delta_velocity = delta_velocities.iter().fold(empty_vector,|acc,v| acc+v);
    let delta_position = delta_positions.iter().fold(empty_vector,|acc,p| acc+p);


    ImuDelta {delta_position,delta_velocity, delta_rotation}
}

fn generate_linear_model_matrices(imu_frame: &ImuDataFrame, delta_rotation_i_k: &Matrix3<Float>, delta_rotation_k: &Matrix3<Float>, gravity_body: &Vector3<Float>) -> (MatrixN<Float,U9>,MatrixMN<Float,U9,U6>) {
    let accel_length = imu_frame.acceleration_count();
    let (_,a_ts_i_k) = imu_frame.acceleration_sublist(0,accel_length-1);

    let gyro_length = imu_frame.gyro_count();
    let (_,g_ts_k) = imu_frame.gyro_sublist(gyro_length-2,gyro_length);

    let a_initial_time_i_k = a_ts_i_k[0];
    let a_delta_t_i_k = a_ts_i_k[1..].iter().map(|t| t - a_initial_time_i_k).fold(0.0, |acc,x| acc+x);
    let a_delta_t_i_k_squared = a_delta_t_i_k.powi(2);

    let g_initial_time_k = g_ts_k[0];
    let g_delta_t_k = g_ts_k[1..].iter().map(|t| t - g_initial_time_k).fold(0.0, |acc,x| acc+x);

    //TODO: compensate for body gravity
    let accelerometer_k = imu_frame.acceleration_data[accel_length-1];
    let accelerometer_skew_symmetric = skew_symmetric(&accelerometer_k);

    let gyrpscope_k = imu_frame.gyro_data[gyro_length-1];
    let right_jacobian = right_jacobian(&gyrpscope_k);

    let identity = Matrix3::<Float>::identity();
    let mut linear_state_design_matrix = MatrixN::<Float,U9>::zeros();
    let mut linear_noise_design_matrix = MatrixMN::<Float,U9,U6>::zeros();

    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(0,0).copy_from(&delta_rotation_k.transpose());
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(3,0).copy_from(&(-a_delta_t_i_k*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(3,3).copy_from(&identity);
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,0).copy_from(&(-(a_delta_t_i_k_squared/2.0)*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,3).copy_from(&(identity*a_delta_t_i_k));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,6).copy_from(&identity);

    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(0,0).copy_from(&(right_jacobian*g_delta_t_k));
    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(3,3).copy_from(&(delta_rotation_i_k*a_delta_t_i_k));
    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(6,3).copy_from(&((a_delta_t_i_k_squared/2.0)*delta_rotation_i_k));

    (linear_state_design_matrix,linear_noise_design_matrix)

}

pub fn propagate_state_covariance(imu_covariance_prev: &ImuCovariance, noise_covariance: &NoiseCovariance, imu_frame: &ImuDataFrame, delta_rotation_i_k: &Matrix3<Float>, delta_rotation_k: &Matrix3<Float>) -> ImuCovariance {

    let (linear_state_design_matrix,linear_noise_design_matrix) = generate_linear_model_matrices(imu_frame, delta_rotation_i_k, delta_rotation_k, &Vector3::<Float>::new(0.0,9.81,0.0));
    linear_state_design_matrix*imu_covariance_prev*linear_state_design_matrix.transpose() + linear_noise_design_matrix*noise_covariance*linear_noise_design_matrix.transpose()
}

pub fn gravityEstimation(data_frames: &Vec<DataFrame>) -> Vector3::<Float> {

    panic!("Gravity Estimation Not yet implemented")
}
