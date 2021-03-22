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
        let mut imu_idx = 0;

        for i in 0..loaded_camera_data.source_timestamps.len() {
            let target_ts = loaded_camera_data.target_timestamps[i];

            let mut imu = ImuDataFrame::from_other(&loaded_imu_data);

            if imu_idx > 0 {
                imu_idx = imu_idx - 1;
            }
            while loaded_imu_data.imu_ts[imu_idx] <= target_ts {
                imu.imu_data.push(loaded_imu_data.imu_data[imu_idx]);
                imu.imu_ts.push(loaded_imu_data.imu_ts[imu_idx]);
                imu_idx += 1;
            }
            imu_sequences.push(imu);
        }

        DataFrame{
            camera_data: loaded_camera_data,
            imu_data_vec: imu_sequences
        }
    }
}