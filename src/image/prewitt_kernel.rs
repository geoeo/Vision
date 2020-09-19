use crate::Float;
use super::kernel::Kernel;


pub struct PrewittKernel {
    kernel: Vec<Float>,
    half_repeat: usize
}

impl PrewittKernel {


    pub fn new() -> PrewittKernel {
        PrewittKernel {
            kernel: vec![-1.0,0.0,1.0],
            half_repeat: 1
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

    fn half_repeat(&self) -> usize {
        self.half_repeat
    }
}
