use crate::{Float,float};
use float::consts::PI;
use super::kernel::Kernel;


pub struct GaussKernel {
    kernel: Vec<Float>,
    step: usize,
}

impl GaussKernel {
    fn sample(mean: Float, std: Float, x:Float) -> Float {
        let exponent = (-0.5*((x-mean)/std).powi(2)).exp();
        let factor = 1.0/(std*(2.0*PI).sqrt());
        factor*exponent
    }

    pub fn new(mean: Float, std: Float, step: usize , end: usize ) -> GaussKernel {
        assert_eq!(end%step,0);

        let start = -(end as isize);
        let end_exclusive = (end as isize) + 1;
        let range = (start..end_exclusive).step_by(step);
        GaussKernel {
            kernel: range.map(|x| GaussKernel::sample(mean,std,x as Float)).collect(),
            step
        }
    }
}

impl Kernel for GaussKernel {
    fn kernel(&self) -> &Vec<Float> {
        &self.kernel
    }

    fn step(&self) -> usize {
        self.step
    }
}



