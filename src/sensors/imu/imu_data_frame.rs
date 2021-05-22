use nalgebra as na;

use na::{Vector3,Vector6,Matrix6};
use crate::odometry::imu_odometry::bias::BiasDelta;
use crate::Float;

#[derive(Clone)]
pub struct ImuDataFrame {
    pub noise_covariance: Matrix6<Float>,
    pub bias_noise_covariance: Matrix6<Float>,
    pub bias_a: Vector3<Float>,
    pub bias_g: Vector3<Float>,
    pub accelerometer_bias_noise_density: Vector3<Float>,
    pub gyro_bias_noise_density: Vector3<Float>,
    pub gyro_data: Vec<Vector3<Float>>,
    pub gyro_ts: Vec<Float>,
    pub acceleration_data: Vec<Vector3<Float>>,
    pub acceleration_ts: Vec<Float>
}

impl ImuDataFrame {


    pub fn empty_from_other(imu_data_frame: &ImuDataFrame) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: imu_data_frame.noise_covariance,
            bias_noise_covariance: imu_data_frame.bias_noise_covariance,
            bias_a: imu_data_frame.bias_a,
            bias_g: imu_data_frame.bias_g,
            accelerometer_bias_noise_density: imu_data_frame.accelerometer_bias_noise_density,
            gyro_bias_noise_density: imu_data_frame.gyro_bias_noise_density,
            acceleration_data: Vec::<Vector3<Float>>::new(),
            acceleration_ts: Vec::<Float>::new(),
            gyro_data: Vec::<Vector3<Float>>::new(),
            gyro_ts: Vec::<Float>::new()
        }

    } 

    pub fn from_data(gyro_data: Vec<Vector3<Float>>,gyro_ts: Vec<Float>, acceleration_data: Vec<Vector3<Float>>,
        acceleration_ts: Vec<Float>, accelerometer_noise_density: Vector3<Float>, gyro_noise_density: Vector3<Float>, 
        accelerometer_bias_noise_density: Vector3<Float>,
        gyro_bias_noise_density: Vector3<Float>,
        accelerometer_bias: Vector3<Float>,
        gyro_bias:Vector3<Float>,
    ) -> ImuDataFrame {
        ImuDataFrame {
            noise_covariance: ImuDataFrame::generate_noise_covariance(&accelerometer_noise_density,&gyro_noise_density),
            bias_noise_covariance: ImuDataFrame::generate_noise_covariance(&accelerometer_bias_noise_density,&gyro_bias_noise_density),
            bias_a: accelerometer_bias,
            bias_g: gyro_bias,
            accelerometer_bias_noise_density: accelerometer_bias_noise_density,
            gyro_bias_noise_density: gyro_bias_noise_density,
            gyro_data,
            gyro_ts,
            acceleration_data,
            acceleration_ts
        }
    }

    pub fn new_from_bias(&self, bias_delta: &BiasDelta) -> ImuDataFrame {

        ImuDataFrame {
            noise_covariance: self.noise_covariance,
            bias_noise_covariance: self.bias_noise_covariance,
            bias_a: self.bias_a + bias_delta.bias_a_delta,
            bias_g: self.bias_g + bias_delta.bias_g_delta,
            accelerometer_bias_noise_density: self.accelerometer_bias_noise_density,
            gyro_bias_noise_density: self.gyro_bias_noise_density,
            gyro_data: self.gyro_data.clone(),
            gyro_ts: self.gyro_ts.clone(),
            acceleration_data: self.acceleration_data.clone(),
            acceleration_ts: self.acceleration_ts.clone()
        }
    }


    fn generate_noise_covariance(accelerometer_noise_density: &Vector3<Float>, gyro_noise_density: &Vector3<Float>) -> Matrix6<Float> {
        let mut noise_covariance = Matrix6::<Float>::zeros();
        noise_covariance[(0,0)] = accelerometer_noise_density[0].powi(2);
        noise_covariance[(1,1)] = accelerometer_noise_density[1].powi(2);
        noise_covariance[(2,2)] = accelerometer_noise_density[2].powi(2);
        noise_covariance[(3,3)] = gyro_noise_density[0].powi(2);
        noise_covariance[(4,4)] = gyro_noise_density[1].powi(2);
        noise_covariance[(5,5)] = gyro_noise_density[2].powi(2);

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