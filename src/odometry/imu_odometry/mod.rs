use nalgebra as na;

use na::{U1,U3,U6,U9,Matrix3,Matrix6,MatrixN,MatrixMN,Vector,Vector3,VectorN,base::storage::Storage};
use crate::Float;
use crate::sensors::{DataFrame, imu::imu_data_frame::ImuDataFrame};
use crate::odometry::imu_odometry::imu_delta::ImuDelta;
use crate::numerics::lie::{exp_r,skew_symmetric,right_jacobian, right_inverse_jacobian, ln_SO3, vector_from_skew_symmetric};

pub mod imu_delta;
pub mod solver;

pub type ImuCovariance = MatrixN<Float,U9>;
pub type ImuResidual = VectorN<Float,U9>;
pub type ImuPertrubation = VectorN<Float,U9>;
pub type NoiseCovariance = Matrix6<Float>;
pub type ImuJacobian = MatrixN<Float,U9>;


#[allow(non_snake_case)]
pub fn pre_integration(imu_data: &ImuDataFrame, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>, gravity_body: &Vector3<Float>) -> (ImuDelta, ImuCovariance) {

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

    let mut imu_covariance = ImuCovariance::zeros();
    for gyro_idx in 0..gyro_delta_times.len() {
        let accelerometer_k = imu_data.acceleration_data[gyro_idx];
        let gyro_k = imu_data.gyro_data[gyro_idx];

        let a_delta_t_i_k = accel_delta_times[0..gyro_idx].iter().fold(0.0,|acc,x| acc+x);
        let g_delta_t_k = gyro_delta_times[0..gyro_idx].iter().fold(0.0,|acc,x| acc+x);

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

fn generate_linear_model_matrices(accelerometer_k: &Vector3<Float>,gyrpscope_k: &Vector3<Float> ,a_delta_t_i_k: Float, g_delta_t_k: Float , delta_rotation_i_k: &Matrix3<Float>, delta_rotation_k: &Matrix3<Float>, gravity_body: &Vector3<Float>) -> (MatrixN<Float,U9>,MatrixMN<Float,U9,U6>) {
    let a_delta_t_i_k_squared = a_delta_t_i_k.powi(2);
    let accelerometer_skew_symmetric = skew_symmetric(&(accelerometer_k + gravity_body));

    let right_jacobian = right_jacobian(&gyrpscope_k);

    let identity = Matrix3::<Float>::identity();
    let mut linear_state_design_matrix = MatrixN::<Float,U9>::zeros();
    let mut linear_noise_design_matrix = MatrixMN::<Float,U9,U6>::zeros();

    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(0,0).copy_from(&identity);
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(0,3).copy_from(&(-(a_delta_t_i_k_squared/2.0)*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(0,6).copy_from(&(identity*a_delta_t_i_k));

    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(3,0).copy_from(&delta_rotation_k.transpose());

    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,3).copy_from(&(-a_delta_t_i_k*delta_rotation_i_k*accelerometer_skew_symmetric));
    linear_state_design_matrix.fixed_slice_mut::<U3,U3>(6,6).copy_from(&identity); 

    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(0,3).copy_from(&((a_delta_t_i_k_squared/2.0)*delta_rotation_i_k));
    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(3,0).copy_from(&(right_jacobian*g_delta_t_k));
    linear_noise_design_matrix.fixed_slice_mut::<U3,U3>(6,3).copy_from(&(delta_rotation_i_k*a_delta_t_i_k)); 

    (linear_state_design_matrix,linear_noise_design_matrix)

}


pub fn generate_jacobian<T>(lie: &Vector<Float,U3,T>, delta_t: Float) -> ImuJacobian where T: Storage<Float,U3,U1> {

    let mut jacobian = ImuJacobian::zeros();
    let identity = Matrix3::<Float>::identity();
    let right_inverse_jacobian = right_inverse_jacobian(&lie);
    jacobian.fixed_slice_mut::<U3,U3>(0,0).copy_from(&identity);
    jacobian.fixed_slice_mut::<U3,U3>(0,6).copy_from(&(-identity*delta_t));
    jacobian.fixed_slice_mut::<U3,U3>(3,3).copy_from(&(right_inverse_jacobian));
    jacobian.fixed_slice_mut::<U3,U3>(6,6).copy_from(&identity);

    jacobian
}

pub fn generate_residual(estimate: &ImuDelta, measurement: &ImuDelta) -> ImuResidual {
    let mut residual = ImuResidual::zeros();
    residual.fixed_rows_mut::<U3>(0).copy_from(&(estimate.delta_position - measurement.delta_position));
    let rotation_residual = measurement.delta_rotation().transpose()*estimate.delta_rotation();
    let w_x = ln_SO3(&rotation_residual);
    residual.fixed_rows_mut::<U3>(3).copy_from(&vector_from_skew_symmetric(&w_x));
    residual.fixed_rows_mut::<U3>(6).copy_from(&(estimate.delta_velocity - measurement.delta_velocity)); 

    residual
}

//TODO
pub fn gravityEstimation(data_frames: &Vec<DataFrame>) -> Vector3::<Float> {

    panic!("Gravity Estimation Not yet implemented")
}
