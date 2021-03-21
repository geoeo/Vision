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

    pub fn sublist<'a>(&'a self, start_index: usize, end_index: usize) -> (&'a [Imu],&'a [Float]) {
        (&self.imu_data[start_index..end_index], &self.imu_ts[start_index..end_index])
    }

    pub fn count(&self) -> usize {
        self.imu_data.len()
    }
}