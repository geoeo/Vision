extern crate image as image_rs;

use crate::image::Image;
use crate::filter::{gauss_kernel::GaussKernel1D,gaussian_2_d_convolution};
use crate::pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters;
use crate::Float;

#[derive(Debug,Clone)]
pub struct OrbOctave {
    pub images: Vec<Image>,
    pub sigmas: Vec<Float>
}

//TODO: Params
impl OrbOctave {
    pub fn build_octave(base_image: &Image,sigma: Float, runtime_parameters: &OrbRuntimeParameters) -> OrbOctave {

        let mean = 0.0;
        let step = 1;

        let blur_radius = runtime_parameters.blur_radius;

        let sigmas = vec!(sigma);
        let kernels: Vec<GaussKernel1D> = sigmas.iter().map(|&sigma| GaussKernel1D::new(mean, sigma,step,blur_radius)).collect();
        let images: Vec<Image> = kernels.iter().map(|kernel| gaussian_2_d_convolution(base_image, kernel, false)).collect();


        OrbOctave{images,sigmas}

    }
}
