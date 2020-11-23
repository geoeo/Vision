extern crate image as image_rs;

use crate::image::Image;
use crate::filter::{gauss_kernel::GaussKernel1D,gaussian_2_d_convolution};
use crate::pyramid::rgbd::rgbd_runtime_parameters::RGBDRuntimeParameters;
use crate::Float;


#[derive(Debug,Clone)]
pub struct RGBDOctave {
    pub gray_images: Vec<Image>,
    pub sigmas: Vec<Float>
}

impl RGBDOctave {

    pub fn build_octave(base_gray_image: &Image, sigma: Float, runtime_parameters: &RGBDRuntimeParameters) -> RGBDOctave {

        let mean = 0.0;
        let step = 1;

        let blur_radius = runtime_parameters.blur_radius;

        let sigmas = vec!(sigma);
        let kernels: Vec<GaussKernel1D> = sigmas.iter().map(|&sigma| GaussKernel1D::new(mean, sigma,step,blur_radius)).collect();
        let gray_images: Vec<Image> = kernels.iter().map(|kernel| gaussian_2_d_convolution(base_gray_image, kernel, true)).collect();


        RGBDOctave{gray_images,sigmas}

    }

}