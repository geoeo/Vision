extern crate nalgebra as na;

use na::DMatrix;
use crate::Float;
use super::kernel::Kernel;


pub struct PrewittKernel {
    kernel: DMatrix<Float>
}

impl PrewittKernel {


    pub fn new() -> PrewittKernel {
        PrewittKernel {
            kernel: DMatrix::from_vec(1,3,vec![-1.0,0.0,1.0])
        }
    }
}

impl Kernel for PrewittKernel {
    fn kernel(&self) -> &DMatrix<Float> {
        &self.kernel
    }

    fn step(&self) -> usize {
        1
    }

    fn normalizing_constant(&self) -> Float{
        2.0
    }
}
