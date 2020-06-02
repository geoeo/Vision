use crate::Float;
use super::kernel::Kernel;


pub struct LaplaceKernel {
    kernel: Vec<Float>
}

impl LaplaceKernel {


    pub fn new() -> LaplaceKernel {
        LaplaceKernel {
            kernel: vec![1.0,-2.0,1.0]
        }
    }
}

impl Kernel for LaplaceKernel {
    fn kernel(&self) -> &Vec<Float> {
        &self.kernel
    }
    fn step(&self) -> usize {
        1
    }
}
