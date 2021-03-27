use crate::sensors::{camera::camera_data_frame::CameraDataFrame, imu::imu_data_frame::ImuDataFrame};

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

        for i in 0..loaded_camera_data.source_timestamps.len() {
            let target_ts = loaded_camera_data.target_timestamps[i];

            let mut imu = ImuDataFrame::empty_from_other(&loaded_imu_data);

            while loaded_imu_data.acceleration_ts[accel_imu_idx] <= target_ts {
                imu.acceleration_data.push(loaded_imu_data.acceleration_data[accel_imu_idx]);
                imu.acceleration_ts.push(loaded_imu_data.acceleration_ts[accel_imu_idx]);
                accel_imu_idx += 1;
            }
            accel_imu_idx = accel_imu_idx - 1;

            while loaded_imu_data.gyro_ts[gyro_imu_idx] <= target_ts {
                imu.gyro_data.push(loaded_imu_data.gyro_data[gyro_imu_idx]);
                imu.gyro_ts.push(loaded_imu_data.gyro_ts[gyro_imu_idx]);
                gyro_imu_idx += 1;
            }
            gyro_imu_idx = gyro_imu_idx - 1;



            imu_sequences.push(imu);
        }

        DataFrame{
            camera_data: loaded_camera_data,
            imu_data_vec: imu_sequences
        }
    }
}