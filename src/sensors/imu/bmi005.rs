extern crate nalgebra as na;

use na::Vector3;
use crate::{float,Float};
use crate::sensors::imu::{imu_data_frame::ImuDataFrame};

const ACCELEROMETER_OUTPUT_SPECTRAL_NOISE_DENSITY: Float = 150.0; //μg/sqrt(Hz)
const ACCELEROMETER_BANDWIDTH: Float = 62.5; // Hz - Realsense specs say 62.5 or 250. 
const ACCELEROMETER_BANDWIDTH_SQRT: Float = 7.90569; // Hz - Sqrt of 62.5
const ACCELEROMETER_WHITE_NOISE_DENSITY: Float = ACCELEROMETER_OUTPUT_SPECTRAL_NOISE_DENSITY*ACCELEROMETER_BANDWIDTH_SQRT; // μg
const ACCELEROMETER_SCALE: Float = 9.81*10e-6;

const GYRO_OUTPUT_SPECTRAL_NOISE_DENSITY: Float = 0.014; //degrees/s/sqrt(Hz)
const GYRO_BANDWIDTH: Float = 200.0;
const GYRO_BANDWIDTH_SQRT:Float = 14.14213562;
const GYRO_WHITE_NOISE_DENSITY: Float = GYRO_OUTPUT_SPECTRAL_NOISE_DENSITY*GYRO_BANDWIDTH_SQRT; // degrees/second 
const GYRO_SCALE: Float = float::consts::PI/180.0;
const BIAS_NOISE: Float = 10e-60; // Bosh Imu seems to be very stable. Drift in specs is given in years


pub fn new_dataframe_from_data(gyro_data: Vec<Vector3<Float>>,gyro_ts: Vec<Float>, accleration_data: Vec<Vector3<Float>>,acceleration_ts: Vec<Float>) -> ImuDataFrame {
    let scaled_acc_white_noise = ACCELEROMETER_WHITE_NOISE_DENSITY*ACCELEROMETER_SCALE;
    let scaled_gyro_white_noise = GYRO_WHITE_NOISE_DENSITY*GYRO_SCALE;

    ImuDataFrame::from_data(gyro_data,gyro_ts,accleration_data,acceleration_ts,scaled_acc_white_noise,scaled_gyro_white_noise)

}

