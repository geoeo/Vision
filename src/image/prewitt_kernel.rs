use crate::Float;
use super::kernel::Kernel;


pub struct PrewittKernel {
    kernel: Vec<Float>
}

impl PrewittKernel {


    pub fn new() -> PrewittKernel {
        PrewittKernel {
            kernel: vec![-1.0,0.0,1.0]
        }
    }
}

impl Kernel for PrewittKernel {
    fn kernel(&self) -> &Vec<Float> {
        &self.kernel
    }

    fn step(&self) -> usize {
        1
    }
}
