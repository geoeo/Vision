use nalgebra as na;

use na::Vector3;
use crate::io::closest_ts_index;
use crate::sensors::{camera::camera_data_frame::CameraDataFrame, imu::imu_data_frame::ImuDataFrame};
use crate::Float;

pub mod camera;
pub mod imu;


pub struct DataFrame {
    pub camera_data: CameraDataFrame,
    pub imu_data_vec: Vec<ImuDataFrame>
}

impl DataFrame {
    pub fn new(loaded_camera_data: CameraDataFrame, loaded_imu_data: ImuDataFrame) -> DataFrame {

        let mut imu_sequences = Vec::<ImuDataFrame>::with_capacity(loaded_camera_data.source_timestamps.len());
        let mut accel_imu_idx = 0;
        let mut gyro_imu_idx = 0;

        while loaded_imu_data.acceleration_ts[accel_imu_idx] <  loaded_camera_data.source_timestamps[0] {
            accel_imu_idx += 1;
        }

        
        while loaded_imu_data.gyro_ts[gyro_imu_idx] <  loaded_camera_data.source_timestamps[0] {
            gyro_imu_idx += 1;
        }

        for i in 0..loaded_camera_data.source_timestamps.len() {

            let target_ts = loaded_camera_data.target_timestamps[i];

            let mut imu = ImuDataFrame::empty_from_other(&loaded_imu_data);
            let mut temp_accel_data = Vec::<Vector3<Float>>::new();
            let mut temp_accel_ts = Vec::<Float>::new();
            let mut accel_hit = false;
            let mut gyro_hit = false;

            while loaded_imu_data.acceleration_ts[accel_imu_idx] <= target_ts {
                

                temp_accel_data.push(loaded_imu_data.acceleration_data[accel_imu_idx]);
                temp_accel_ts.push(loaded_imu_data.acceleration_ts[accel_imu_idx]);

                accel_imu_idx += 1;
                accel_hit = true;
            }
            if accel_hit {
                accel_imu_idx -= 1;
            }


            //TODO: use spline interpolation
            while loaded_imu_data.gyro_ts[gyro_imu_idx] <= target_ts {
                imu.gyro_data.push(loaded_imu_data.gyro_data[gyro_imu_idx]);
                imu.gyro_ts.push(loaded_imu_data.gyro_ts[gyro_imu_idx]);

                imu.acceleration_ts.push(loaded_imu_data.gyro_ts[gyro_imu_idx]);
                let accel_idx = closest_ts_index(loaded_imu_data.gyro_ts[gyro_imu_idx],&temp_accel_ts);
                imu.acceleration_data.push(loaded_imu_data.acceleration_data[accel_idx]);

            
                gyro_imu_idx += 1;
                gyro_hit = true
            }

            if gyro_hit {
                gyro_imu_idx -= 1;
            }



            assert_eq!(imu.acceleration_ts.len(),imu.gyro_ts.len());
            assert_eq!(imu.acceleration_data.len(),imu.gyro_data.len());

            if imu.acceleration_ts.len() == 0 {
                println!("warning no data!");
            }


            imu_sequences.push(imu);


        }

        DataFrame{
            camera_data: loaded_camera_data,
            imu_data_vec: imu_sequences
        }
    }
}