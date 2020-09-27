extern crate nalgebra as na;

use na::DMatrix;
use crate::Float;
use super::kernel::Kernel;


pub struct LaplaceKernel {
    kernel: DMatrix<Float>
}

impl LaplaceKernel {


    pub fn new() -> LaplaceKernel {
        LaplaceKernel {
            kernel: DMatrix::from_vec(1,3,vec![1.0,-2.0,1.0])
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

    fn normalizing_constant(&self) -> Float{
        1.0
    }
}
