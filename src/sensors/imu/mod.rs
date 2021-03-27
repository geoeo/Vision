// use nalgebra as na;

// use na::Vector3;
// use crate::Float;

pub mod imu_data_frame;
pub mod bmi005;

// //TODO: bias -> could even put this into data frame as assumption is that it remains constants between two keyframes
// #[derive(Debug,Copy,Clone)]
// pub struct Imu {
//     pub accelerometer: Vector3<Float>,
//     pub gyro: Vector3<Float>
// }

// impl Imu {
//     //TODO: these are note 1:1 change
//     pub fn new(accelerometer: &Vec<Float>, gyro: &Vec<Float>,acc_scale:Float,gyro_scale:Float) -> Imu {
//         assert_eq!(accelerometer.len(),3);
//         assert_eq!(gyro.len(),3);

//         let scaled_accelerometer = Vector3::<Float>::new(accelerometer[0],accelerometer[1],accelerometer[2])*acc_scale;
//         let scaled_gyro = Vector3::<Float>::new(gyro[0],gyro[1],gyro[2])*gyro_scale;

//         Imu{accelerometer: scaled_accelerometer, gyro: scaled_gyro}
//     }

// }