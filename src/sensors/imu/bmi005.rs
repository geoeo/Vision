extern crate nalgebra as na;

use na::Vector3;
use crate::{float,Float};
use crate::sensors::imu::{imu_data_frame::ImuDataFrame};

const ACCELEROMETER_OUTPUT_SPECTRAL_NOISE_DENSITY: Float = 150.0; //μg/sqrt(Hz)
const ACCELEROMETER_BANDWIDTH: Float = 62.5; // Hz - Realsense specs say 62.5.
const ACCELEROMETER_BANDWIDTH_SQRT: Float = 7.90569; // Hz - Sqrt of 62.5
const ACCELEROMETER_WHITE_NOISE_DENSITY: Float = ACCELEROMETER_OUTPUT_SPECTRAL_NOISE_DENSITY*ACCELEROMETER_BANDWIDTH_SQRT; // μg
const ACCELEROMETER_SCALE: Float = 9.81*10e-6;
const ACCELEROMETER_BIAS_SPECTRAL_NOISE_DENSITY: Float = ACCELEROMETER_OUTPUT_SPECTRAL_NOISE_DENSITY/10.0; // TODO: Calibrate this is just a heuristic

const GYRO_OUTPUT_SPECTRAL_NOISE_DENSITY: Float = 0.014; //degrees/s/sqrt(Hz)
const GYRO_BANDWIDTH: Float = 200.0;
const GYRO_BANDWIDTH_SQRT:Float = 14.14213562; // Sqrt of 200
const GYRO_WHITE_NOISE_DENSITY: Float = GYRO_OUTPUT_SPECTRAL_NOISE_DENSITY*GYRO_BANDWIDTH_SQRT; // degrees/second 
const GYRO_SCALE: Float = float::consts::PI/180.0;
const GYRO_BIAS_SPECTRAL_NOISE_DENSITY: Float = GYRO_OUTPUT_SPECTRAL_NOISE_DENSITY/10.0; // TODO: Calibrate this is just a heuristic


pub fn new_dataframe_from_data(gyro_data: Vec<Vector3<Float>>,gyro_ts: Vec<Float>, accleration_data: Vec<Vector3<Float>>,acceleration_ts: Vec<Float>) -> ImuDataFrame {
    let scaled_acc_spectral_white_noise = ACCELEROMETER_WHITE_NOISE_DENSITY*ACCELEROMETER_SCALE;
    let scaled_gyro_spectral_white_noise = GYRO_WHITE_NOISE_DENSITY*GYRO_SCALE;

    let scaled_acc_bias_white_noise = ACCELEROMETER_BANDWIDTH_SQRT*ACCELEROMETER_BIAS_SPECTRAL_NOISE_DENSITY*ACCELEROMETER_SCALE; 
    let scaled_gyro_bias_white_noise = GYRO_BANDWIDTH_SQRT*GYRO_BIAS_SPECTRAL_NOISE_DENSITY*GYRO_SCALE;

    //TODO: calibrate imu
    ImuDataFrame::from_data(gyro_data,gyro_ts,
        accleration_data,
        acceleration_ts,
        Vector3::<Float>::new(scaled_acc_spectral_white_noise,scaled_acc_spectral_white_noise,scaled_acc_spectral_white_noise),
        Vector3::<Float>::new(scaled_gyro_spectral_white_noise,scaled_gyro_spectral_white_noise,scaled_gyro_spectral_white_noise),
        //Vector3::<Float>::new(10e1*scaled_acc_bias_white_noise,scaled_acc_bias_white_noise,70e5*scaled_acc_bias_white_noise),
        Vector3::<Float>::new(10e1*scaled_acc_bias_white_noise,scaled_acc_bias_white_noise,30e4*scaled_acc_bias_white_noise),
        Vector3::<Float>::new(scaled_gyro_bias_white_noise,scaled_gyro_bias_white_noise,scaled_gyro_bias_white_noise),
        Vector3::<Float>::new(0.6,0.2,0.0), // simple from plots
        Vector3::<Float>::new(0.0,0.0,0.0),
    )

}

