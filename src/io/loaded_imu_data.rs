use crate::Float;
use crate::imu::Imu;

#[derive(Clone)]
pub struct LoadedImuData {
    pub imu_data: Vec<Imu>,
    pub imu_ts: Vec<Float>
}