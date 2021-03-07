use crate::{float,Float};
use crate::sensors::imu::Imu;

const ACCELEROMETER_OUTPUT_NOISE_DENSITY: Float = 150.0; //μg/sqrt(Hz)
const ACCELEROMETER_BANDWIDTH: Float = 250.0; // Hz - Realsense specs say 62.5 or 250. 
const ACCELEROMETER_BANDWIDTH_SQRT: Float = 15.8113883; // Hz - Sqrt of 250
const ACCELEROMETER_WHITE_NOISE: Float = ACCELEROMETER_OUTPUT_NOISE_DENSITY*ACCELEROMETER_BANDWIDTH_SQRT; // μg
const ACCELEROMETER_SCALE: Float = 9.81*10e-6;
const GYRO_WHITE_NOISE: Float = 0.1; // degrees/second @ BW=47 HZ
const GYRO_SCALE: Float = float::consts::PI/180.0;
const BIAS_NOISE: Float = 10e-60; // Bosh Imu seems to be very stable. Drift in specs is given in years


pub fn new(accelerometer: &Vec<Float>, gyro: &Vec<Float>) -> Imu {
    Imu::new(accelerometer, gyro, ACCELEROMETER_WHITE_NOISE,ACCELEROMETER_SCALE,GYRO_WHITE_NOISE,GYRO_SCALE)
}

