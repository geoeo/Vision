use crate::Float;
use crate::sensors::imu::Imu;

#[derive(Clone)]
pub struct ImuDataFrame {
    pub imu_data: Vec<Imu>,
    pub imu_ts: Vec<Float>
}

impl ImuDataFrame {
    pub fn new() -> ImuDataFrame {
        ImuDataFrame {
            imu_data: Vec::<Imu>::new(),
            imu_ts: Vec::<Float>::new()
        }
    }
}