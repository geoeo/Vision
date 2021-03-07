use nalgebra as na;

use na::{Matrix3,Vector3};
use crate::Float;
use crate::sensors::{DataFrame,imu::imu_data_frame::ImuDataFrame};
use crate::odometry::imu_odometry::imu_measurement::{ImuState,ImuCovariance};
use crate::numerics::lie::exp_r;

pub mod imu_measurement;


#[allow(non_snake_case)]
pub fn PreIntegration(imu_data: &ImuDataFrame, bias_gyroscope: &Vector3<Float>,bias_accelerometer: &Vector3<Float>) -> ImuState {

    let initial_time = imu_data.imu_ts[0];
    let delta_times = imu_data.imu_ts[1..].iter().map(|t| t - initial_time).collect::<Vec<Float>>();

    let delta_rotations = imu_data.imu_data[1..].iter().zip(delta_times.iter()).map(|(x,&dt)| (x.gyro-bias_gyroscope)*dt).map(|x| exp_r(&x)).collect::<Vec<Matrix3::<Float>>>();
    let delta_velocities = imu_data.imu_data[1..].iter().zip(delta_rotations.iter()).zip(delta_times.iter()).map(|((x,dR),&dt)| dR*(x.accelerometer - bias_accelerometer)*dt).collect::<Vec<Vector3<Float>>>();
    let delta_positions = delta_velocities.iter().zip(delta_times.iter()).map(|(dv,&dt)| dv*dt+0.5*dv*dt).collect::<Vec<Vector3::<Float>>>(); // TODO: check this simplification

    let identity = Matrix3::<Float>::identity();
    let empty_vector = Vector3::<Float>::identity();

    let delta_rotation = delta_rotations.iter().fold(identity,|acc,x| acc*x);
    let delta_velocity = delta_velocities.iter().fold(empty_vector,|acc,v| acc+v);
    let delta_position = delta_positions.iter().fold(empty_vector,|acc,p| acc+p);

    ImuState {position: delta_position,velocity: delta_velocity, orientation: delta_rotation}

}

pub fn GravityEstimation(data_frames: &Vec<DataFrame>) -> Vector3::<Float> {

    panic!("Gravity Estimation Not yet implemented")
}