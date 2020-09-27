extern crate nalgebra as na;

use na::DMatrix;
use crate::{Float,float};
use float::consts::PI;
use super::kernel::Kernel;


pub struct GaussKernel1D {
    kernel: DMatrix<Float>,
    step: usize,
}

impl GaussKernel1D {
    fn sample(mean: Float, std: Float, x:Float) -> Float {
        let exponent = (-0.5*((x-mean)/std).powi(2)).exp();
        let factor = 1.0/(std*(2.0*PI).sqrt());
        factor*exponent
    }

    pub fn new(mean: Float, std: Float, step: usize , half_width: Float ) -> GaussKernel1D {
        let half_width_usize = half_width.trunc() as usize;
        assert_eq!(half_width_usize%step,0);

        let cols = 2*half_width_usize+1;
        let start = -(half_width_usize as isize);
        let end_exclusive = (half_width as isize) + 1;
        let range = (start..end_exclusive).step_by(step);
        GaussKernel1D {
            kernel: DMatrix::from_vec(1,cols,range.map(|x| GaussKernel1D::sample(mean,std,x as Float)).collect()),
            step
        }
    }
}

impl Kernel for GaussKernel1D {
    fn kernel(&self) -> &DMatrix<Float> {
        &self.kernel
    }

    fn step(&self) -> usize {
        self.step
    }

    fn half_repeat(&self) -> usize {
        0
    }

    fn normalizing_constant(&self) -> Float{
        1.0
    }
}



