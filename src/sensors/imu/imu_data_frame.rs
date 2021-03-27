use nalgebra as na;

use na::{Vector3,Matrix6};
use crate::Float;

#[derive(Clone)]
pub struct ImuDataFrame {
    pub noise_covariance: Matrix6<Float>,
    pub gyro_data: Vec<Vector3<Float>>,
    pub gyro_ts: Vec<Float>,
    pub acceleration_data: Vec<Vector3<Float>>,
    pub acceleration_ts: Vec<Float>
}

impl ImuDataFrame {
    pub fn new(acceleration_inital_capacity: usize, gyro_inital_capacity: usize, accelerometer_noise_density: Float, gyro_noise_density:Float) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: ImuDataFrame::generate_noise_covariance(accelerometer_noise_density,gyro_noise_density),
            acceleration_data: Vec::<Vector3<Float>>::with_capacity(acceleration_inital_capacity),
            acceleration_ts: Vec::<Float>::with_capacity(acceleration_inital_capacity),
            gyro_data: Vec::<Vector3<Float>>::with_capacity(gyro_inital_capacity),
            gyro_ts: Vec::<Float>::with_capacity(gyro_inital_capacity)
        }
    }

    pub fn empty_from_other(imu_data_frame: &ImuDataFrame) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: imu_data_frame.noise_covariance,
            acceleration_data: Vec::<Vector3<Float>>::new(),
            acceleration_ts: Vec::<Float>::new(),
            gyro_data: Vec::<Vector3<Float>>::new(),
            gyro_ts: Vec::<Float>::new()
        }

    }

    pub fn from_data(gyro_data: Vec<Vector3<Float>>,gyro_ts: Vec<Float>, acceleration_data: Vec<Vector3<Float>>,acceleration_ts: Vec<Float>, accelerometer_noise_density: Float, gyro_noise_density:Float) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: ImuDataFrame::generate_noise_covariance(accelerometer_noise_density,gyro_noise_density),
            gyro_data,
            gyro_ts,
            acceleration_data,
            acceleration_ts
        }
    }

    fn generate_noise_covariance( accelerometer_noise_density: Float, gyro_noise_density:Float) -> Matrix6<Float> {
        let mut noise_covariance = Matrix6::<Float>::zeros();
        noise_covariance[(0,0)] = gyro_noise_density;
        noise_covariance[(1,1)] = gyro_noise_density;
        noise_covariance[(2,2)] = gyro_noise_density;
        noise_covariance[(3,3)] = accelerometer_noise_density;
        noise_covariance[(4,4)] = accelerometer_noise_density;
        noise_covariance[(5,5)] = accelerometer_noise_density;

        noise_covariance
    }

    pub fn acceleration_sublist<'a>(&'a self, start_index: usize, end_index: usize) -> (&'a [Vector3<Float>],&'a [Float]) {
        (&self.acceleration_data[start_index..end_index], &self.acceleration_ts[start_index..end_index])
    }

    pub fn gyro_sublist<'a>(&'a self, start_index: usize, end_index: usize) -> (&'a [Vector3<Float>],&'a [Float]) {
        (&self.gyro_data[start_index..end_index], &self.gyro_ts[start_index..end_index])
    }

    pub fn acceleration_count(&self) -> usize {
        self.acceleration_ts.len()
    }

    pub fn gyro_count(&self) -> usize {
        self.gyro_ts.len()
    }
}