use nalgebra as na;

use na::Vector3;
use crate::Float;
use crate::sensors::DataFrame;
use crate::odometry::imu_odometry::{imu_state::ImuState,imu_noise_state::ImuNoiseState};


pub mod imu_noise_state;
pub mod imu_state;



pub fn PreIntegration(data_frame: &DataFrame) -> (ImuState, ImuNoiseState) {


    panic!("PreIntegration not yet implemented")

}

pub fn GravityEstimation(data_frame: &DataFrame) -> Vector3::<Float> {

    panic!("Gravity Estimation Not yet implemented")
}