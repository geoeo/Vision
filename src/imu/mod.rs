use nalgebra as na;

use na::Vector3;
use crate::Float;

#[derive(Debug,Copy,Clone)]
pub struct Imu {
    pub accelerometer: Vector3<Float>,
    pub gyro: Vector3<Float>,
    pub acc_white_noise: Float,
    pub gyro_white_noise: Float
}

impl Imu {
    pub fn new(accelerometer: &Vec<Float>, gyro: &Vec<Float>,acc_white_noise: Float,acc_scale: Float, gyro_white_noise: Float, gyro_scale: Float) -> Imu {
        assert_eq!(accelerometer.len(),3);
        assert_eq!(gyro.len(),3);

        let scaled_accelerometer = Vector3::<Float>::new(accelerometer[0],accelerometer[1],accelerometer[2])*acc_scale;
        let scaled_gyro = Vector3::<Float>::new(gyro[0],gyro[1],gyro[2])*gyro_scale;
        let scaled_acc_white_noise = acc_white_noise*acc_scale;
        let scaled_gyro_white_noise = gyro_white_noise*gyro_scale;

        Imu{accelerometer: scaled_accelerometer, 
            gyro: scaled_gyro,
            acc_white_noise: scaled_acc_white_noise,
            gyro_white_noise: scaled_gyro_white_noise}
    }

}