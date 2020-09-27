extern crate nalgebra as na;

use na::DMatrix;
use crate::Float;
use super::kernel::Kernel;


pub struct LaplaceKernel {
    kernel: DMatrix<Float>, 
    half_repeat: usize

}

impl LaplaceKernel {


    pub fn new() -> LaplaceKernel {
        LaplaceKernel {
            kernel: DMatrix::from_vec(1,3,vec![1.0,-2.0,1.0]),
            half_repeat: 1
        }
    }
}

impl Kernel for LaplaceKernel {
    fn kernel(&self) -> &DMatrix<Float> {
        &self.kernel
    }
    fn step(&self) -> usize {
        1
    }
    fn half_repeat(&self) -> usize {
        self.half_repeat
    }

    fn normalizing_constant(&self) -> Float{
        1.0
    }
}
