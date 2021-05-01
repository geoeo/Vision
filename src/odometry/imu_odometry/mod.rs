use nalgebra as na;

use na::{U1,U3,U6,U9,Matrix3,SMatrix,SVector,Vector,Vector3,UnitQuaternion,base::storage::Storage};
use crate::{float,Float};
use crate::sensors::{DataFrame, imu::imu_data_frame::ImuDataFrame};
use crate::odometry::imu_odometry::imu_delta::ImuDelta;
use crate::numerics::lie::{exp_r,skew_symmetric,right_jacobian, right_inverse_jacobian, ln_SO3, vector_from_skew_symmetric};

pub mod imu_delta;
pub mod solver;

pub type ImuCovariance = SMatrix<Float,9,9>;
pub type ImuResidual = SVector<Float,9>;
pub type ImuPertrubation = SVector<Float,9>;
pub type NoiseCovariance = SMatrix<Float,6,6>;
pub type ImuJacobian = SMatrix<Float,9,9>;


#[allow(non_snake_case)]
pub fn pre_integration(imu_data: &ImuDataFrame, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>) -> (ImuDelta, ImuCovariance) {

    let accel_delta_times = imu_data.acceleration_ts[1..].iter().enumerate().map(|(i,t)| t - imu_data.acceleration_ts[i]).collect::<Vec<Float>>();
    let gyro_delta_times = imu_data.gyro_ts[1..].iter().enumerate().map(|(i,t)| t - imu_data.gyro_ts[i]).collect::<Vec<Float>>();
    
    let delta_rotations = imu_data.gyro_data[0..imu_data.gyro_count()-1].iter().zip(gyro_delta_times.iter()).map(|(x,&dt)| (x-bias_gyroscope)*dt).map(|x| exp_r(&x)).collect::<Vec<Matrix3::<Float>>>();
    let delta_velocities = imu_data.acceleration_data[0..imu_data.acceleration_count()-1].iter().zip(delta_rotations.iter()).zip(accel_delta_times.iter()).map(|((x,dR),&dt)| dR*(x - bias_accelerometer + gravity_body)*dt).collect::<Vec<Vector3<Float>>>(); 
    let accumulated_velocities = delta_velocities.iter().scan(Vector3::<Float>::zeros(),|acc,dv| {
        *acc=*acc+*dv;
        Some(*acc)
    }).collect::<Vec<Vector3<Float>>>();
    let delta_positions = delta_velocities.iter().zip(accel_delta_times.iter()).zip(accumulated_velocities.iter()).map(|((dv,&dt),v_initial)| v_initial*dt +0.5*dv*dt).collect::<Vec<Vector3::<Float>>>(); 


    let identity = Matrix3::<Float>::identity();
    let empty_vector = Vector3::<Float>::zeros();

    let mut imu_covariance = ImuCovariance::zeros();
    for gyro_idx in 0..gyro_delta_times.len() {
        let accelerometer_k = imu_data.acceleration_data[gyro_idx];
        let gyro_k = imu_data.gyro_data[gyro_idx];

        let a_delta_t_i_k = accel_delta_times[0..gyro_idx].iter().fold(0.0,|acc,x| acc+x);
        let g_delta_t_k = gyro_delta_times[gyro_idx];

        let delta_rotation_i_k = delta_rotations[0..gyro_idx].iter().fold(identity,|acc,x| acc*x);
        let delta_rotation_k = delta_rotations[gyro_idx];

        let (linear_state_design_matrix,linear_noise_design_matrix) = generate_linear_model_matrices(&accelerometer_k, &gyro_k,a_delta_t_i_k,g_delta_t_k ,&delta_rotation_i_k, &delta_rotation_k, gravity_body);
        imu_covariance = linear_state_design_matrix*imu_covariance*linear_state_design_matrix.transpose() + linear_noise_design_matrix*imu_data.noise_covariance*linear_noise_design_matrix.transpose()

    }

    let number_of_rotations = delta_rotations.len();
    let delta_rotation_i_k = delta_rotations[0..number_of_rotations-1].iter().fold(identity,|acc,x| acc*x);
    let delta_rotation_k = delta_rotations[number_of_rotations-1];

    let delta_velocity = delta_velocities.iter().fold(empty_vector,|acc,v| acc+v);
    let delta_position = delta_positions.iter().fold(empty_vector,|acc,p| acc+p);


    (ImuDelta {delta_position,delta_velocity, delta_rotation_i_k,delta_rotation_k}, imu_covariance)
}

fn generate_linear_model_matrices(accelerometer_k: &Vector3<Float>,gyrpscope_k: &Vector3<Float> ,a_delta_t_i_k: Float, g_delta_t_k: Float , delta_rotation_i_k: &Matrix3<Float>, delta_rotation_k: &Matrix3<Float>, gravity_body: &Vector3<Float>) -> (SMatrix<Float,9,9>,SMatrix<Float,9,6>) {
    let a_delta_t_i_k_squared = a_delta_t_i_k.powi(2);
    let accelerometer_skew_symmetric = skew_symmetric(&(accelerometer_k + gravity_body));

    let right_jacobian = right_jacobian(&gyrpscope_k);

    let identity = Matrix3::<Float>::identity();
    let mut linear_state_design_matrix = SMatrix::<Float,9,9>::zeros();
    let mut linear_noise_design_matrix = SMatrix::<Float,9,6>::zeros();

    linear_state_design_matrix.fixed_slice_mut::<3,3>(0,0).copy_from(&identity);
    linear_state_design_matrix.fixed_slice_mut::<3,3>(0,3).copy_from(&(-(a_delta_t_i_k_squared/2.0)*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<3,3>(0,6).copy_from(&(identity*a_delta_t_i_k));

    linear_state_design_matrix.fixed_slice_mut::<3,3>(3,0).copy_from(&delta_rotation_k.transpose());

    linear_state_design_matrix.fixed_slice_mut::<3,3>(6,3).copy_from(&(-a_delta_t_i_k*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<3,3>(6,6).copy_from(&identity); 

    linear_noise_design_matrix.fixed_slice_mut::<3,3>(0,3).copy_from(&((a_delta_t_i_k_squared/2.0)*delta_rotation_i_k));
    linear_noise_design_matrix.fixed_slice_mut::<3,3>(3,0).copy_from(&(right_jacobian*g_delta_t_k));
    linear_noise_design_matrix.fixed_slice_mut::<3,3>(6,3).copy_from(&(delta_rotation_i_k*a_delta_t_i_k)); 

    (linear_state_design_matrix,linear_noise_design_matrix)

}


pub fn generate_jacobian<R, const T: usize>(lie: &Vector<Float,U3,R>, delta_t: Float) -> SMatrix<Float,T,T> where R: Storage<Float,U3,U1> {

    let mut jacobian = SMatrix::<Float,T,T>::zeros();
        //TODO: check if T >= 9
    let identity = Matrix3::<Float>::identity();
    let right_inverse_jacobian = right_inverse_jacobian(&lie);
    jacobian.fixed_slice_mut::<3,3>(0,0).copy_from(&identity);
    jacobian.fixed_slice_mut::<3,3>(0,6).copy_from(&(-identity*delta_t));
    jacobian.fixed_slice_mut::<3,3>(3,3).copy_from(&(right_inverse_jacobian));
    jacobian.fixed_slice_mut::<3,3>(6,6).copy_from(&identity);

    jacobian
}

pub fn generate_residual<const T: usize>(estimate: &ImuDelta, measurement: &ImuDelta) -> SVector<Float,T> {
    let mut residual = SVector::<Float,T>::zeros();
    //TODO: check if T >= 9
    residual.fixed_rows_mut::<3>(0).copy_from(&(estimate.delta_position - measurement.delta_position));
    let rotation_residual = measurement.delta_rotation().transpose()*estimate.delta_rotation();
    let w_x = ln_SO3(&rotation_residual);
    residual.fixed_rows_mut::<3>(3).copy_from(&vector_from_skew_symmetric(&w_x));
    residual.fixed_rows_mut::<3>(6).copy_from(&(estimate.delta_velocity - measurement.delta_velocity)); 

    residual
}

pub fn weight_residuals<const T : usize>(residual: &mut SVector<Float, T>, weights: &SMatrix<Float,T,T>) -> () {
    weights.mul_to(&residual.clone(),residual);
}

pub fn weight_jacobian<const T: usize>(jacobian: &mut SMatrix<Float,T,T>, weights: &SMatrix<Float,T,T>) -> () {
    weights.mul_to(&jacobian.clone(),jacobian);
}

//TODO
pub fn gravityEstimation(data_frames: &Vec<DataFrame>) -> Vector3::<Float> {

    panic!("Gravity Estimation Not yet implemented")
}

