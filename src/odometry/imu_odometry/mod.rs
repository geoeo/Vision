use nalgebra as na;

use na::{U3,U6,U9,Matrix3,MatrixN,MatrixMN,Vector3};
use crate::Float;
use crate::sensors::{DataFrame,imu::imu_data_frame::ImuDataFrame};
use crate::odometry::imu_odometry::imu_measurement::{ImuState,ImuCovariance, NoiseCovariance};
use crate::numerics::lie::{exp_r,skew_symmetric,right_jacobian};

pub mod imu_measurement;


#[allow(non_snake_case)]
pub fn pre_integration(imu_data: &ImuDataFrame, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>) -> ImuState {

    let initial_time = imu_data.imu_ts[0];
    let initial_acceleration = imu_data.imu_data[0].accelerometer;
    let delta_times = imu_data.imu_ts[1..].iter().map(|t| t - initial_time).collect::<Vec<Float>>();

    let delta_rotations = imu_data.imu_data[1..].iter().zip(delta_times.iter()).map(|(x,&dt)| (x.gyro-bias_gyroscope)*dt).map(|x| exp_r(&x)).collect::<Vec<Matrix3::<Float>>>();
    let delta_velocities = imu_data.imu_data[1..].iter().zip(delta_rotations.iter()).zip(delta_times.iter()).map(|((x,dR),&dt)| dR*(x.accelerometer - bias_accelerometer)*dt).collect::<Vec<Vector3<Float>>>();
    let delta_positions = delta_velocities.iter().zip(delta_times.iter()).map(|(dv,&dt)| 1.5*dv*dt).collect::<Vec<Vector3::<Float>>>(); //TODO: Check this

    let identity = Matrix3::<Float>::identity();
    let empty_vector = Vector3::<Float>::identity();

    let delta_rotation = delta_rotations.iter().fold(identity,|acc,x| acc*x);
    let delta_velocity = delta_velocities.iter().fold(empty_vector,|acc,v| acc+v);
    let delta_position = delta_positions.iter().fold(empty_vector,|acc,p| acc+p);

    ImuState {position: delta_position,velocity: delta_velocity, orientation: delta_rotation}
}

fn generate_linear_model_matrices(imu_frame: &ImuDataFrame, delta_rotation_i_k: &Matrix3<Float>, delta_rotation_k: &Matrix3<Float>) -> (MatrixN<Float,U9>,MatrixMN<Float,U9,U6>) {
    let length = imu_frame.count();
    let (_,ts_i_k) = imu_frame.sublist(0,length-1);
    let (_,ts_k) = imu_frame.sublist(length-2,length);

    let initial_time_i_k = ts_i_k[0];
    let delta_t_i_k = ts_i_k[1..].iter().map(|t| t - initial_time_i_k).fold(0.0, |acc,x| acc+x);
    let delta_t_i_k_squared = delta_t_i_k.powi(2);

    let initial_time_k = ts_i_k[0];
    let delta_t_k = ts_k[1..].iter().map(|t| t - initial_time_k).fold(0.0, |acc,x| acc+x);

    let accelerometer_k = imu_frame.imu_data[length-1].accelerometer;
    let accelerometer_skew_symmetric = skew_symmetric(&accelerometer_k);

    let gyrpscope_k = imu_frame.imu_data[length-1].gyro;
    let right_jacobian = right_jacobian(&gyrpscope_k);

    let identity = Matrix3::<Float>::identity();
    let mut linear_state_design_matrix = MatrixN::<Float,U9>::zeros();
    let mut linear_noise_design_matrix = MatrixMN::<Float,U9,U6>::zeros();

    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(0,0).copy_from(&delta_rotation_k.transpose());
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(3,0).copy_from(&(-delta_t_i_k*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(3,3).copy_from(&identity);
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,0).copy_from(&(-(delta_t_i_k_squared/2.0)*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,3).copy_from(&(identity*delta_t_i_k));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,6).copy_from(&identity);

    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(0,0).copy_from(&(right_jacobian*delta_t_k));
    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(3,3).copy_from(&(delta_rotation_i_k*delta_t_i_k));
    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(6,3).copy_from(&((delta_t_i_k_squared/2.0)*delta_rotation_i_k));

    (linear_state_design_matrix,linear_noise_design_matrix)

}

pub fn propagate_state_covariance(imu_covariance_prev: &ImuCovariance, noise_covariance: &NoiseCovariance, imu_frame: &ImuDataFrame, delta_rotation_i_k: &Matrix3<Float>, delta_rotation_k: &Matrix3<Float>) -> ImuCovariance {

    let (linear_state_design_matrix,linear_noise_design_matrix) = generate_linear_model_matrices(imu_frame, delta_rotation_i_k, delta_rotation_k);
    linear_state_design_matrix*imu_covariance_prev*linear_state_design_matrix.transpose() + linear_noise_design_matrix*noise_covariance*linear_noise_design_matrix.transpose()
}

pub fn gravityEstimation(data_frames: &Vec<DataFrame>) -> Vector3::<Float> {

    panic!("Gravity Estimation Not yet implemented")
}
