extern crate nalgebra as na;

use na::DMatrix;
use crate::Float;
use super::kernel::Kernel;


pub struct LaplaceOffCenterKernel {
    kernel: DMatrix<Float>
}

//TODO: this is unused
impl LaplaceOffCenterKernel {


    pub fn new() -> LaplaceOffCenterKernel {
        LaplaceOffCenterKernel {
            kernel: DMatrix::from_vec(1,3,vec![-1.0,0.0,1.0])
        }
    }
}

impl Kernel for LaplaceOffCenterKernel {
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
