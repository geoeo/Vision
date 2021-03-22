use nalgebra as na;

use na::Matrix6;
use crate::Float;
use crate::sensors::imu::Imu;

#[derive(Clone)]
pub struct ImuDataFrame {
    pub noise_covariance: Matrix6<Float>,
    pub imu_data: Vec<Imu>,
    pub imu_ts: Vec<Float>
}

impl ImuDataFrame {
    pub fn new(inital_capacity: usize, accelerometer_noise_density: Float, gyro_noise_density:Float) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: ImuDataFrame::generate_noise_covariance(accelerometer_noise_density,gyro_noise_density),
            imu_data: Vec::<Imu>::with_capacity(inital_capacity),
            imu_ts: Vec::<Float>::with_capacity(inital_capacity)
        }
    }

    pub fn from_other(imu_data_frame: &ImuDataFrame) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: imu_data_frame.noise_covariance,
            imu_data: Vec::<Imu>::new(),
            imu_ts: Vec::<Float>::new()
        }

    }

    pub fn from_data(imu_data: Vec<Imu>,imu_ts: Vec<Float>, accelerometer_noise_density: Float, gyro_noise_density:Float) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: ImuDataFrame::generate_noise_covariance(accelerometer_noise_density,gyro_noise_density),
            imu_data,
            imu_ts
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

    pub fn sublist<'a>(&'a self, start_index: usize, end_index: usize) -> (&'a [Imu],&'a [Float]) {
        (&self.imu_data[start_index..end_index], &self.imu_ts[start_index..end_index])
    }

    pub fn count(&self) -> usize {
        self.imu_data.len()
    }
}