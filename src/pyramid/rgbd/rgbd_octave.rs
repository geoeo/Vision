extern crate image as image_rs;

use crate::image::Image;
use crate::filter::{prewitt_kernel::PrewittKernel,gauss_kernel::GaussKernel1D,gaussian_2_d_convolution,filter_1d_convolution};
use crate::pyramid::rgbd::rgbd_runtime_parameters::RGBDRuntimeParameters;
use crate::{Float,GradientDirection};


#[derive(Debug,Clone)]
pub struct RGBDOctave {
    pub gray_images: Vec<Image>,
    pub x_gradients: Vec<Image>,
    pub y_gradients: Vec<Image>,
    pub sigmas: Vec<Float>
}

impl RGBDOctave {

    pub fn build_octave(base_gray_image: &Image, sigma: Float, runtime_parameters: &RGBDRuntimeParameters) -> RGBDOctave {

        let mean = 0.0;
        let step = 1;

        let blur_radius = runtime_parameters.blur_radius;
        let prewitt_kernel = PrewittKernel::new();

        let sigmas = vec!(sigma);
        let kernels: Vec<GaussKernel1D> = sigmas.iter().map(|&sigma| GaussKernel1D::new(mean, sigma,step,blur_radius)).collect();
        let gray_images: Vec<Image> = kernels.iter().map(|kernel| gaussian_2_d_convolution(base_gray_image, kernel, true)).collect();
        let images_borrows: Vec<&Image> = gray_images.iter().map(|x| x).collect();
        let x_gradients: Vec<Image> = (0..sigmas.len()).map(|x| filter_1d_convolution(&images_borrows,x, GradientDirection::HORIZINTAL, &prewitt_kernel, false)).collect();
        let y_gradients: Vec<Image> = (0..sigmas.len()).map(|x| filter_1d_convolution(&images_borrows,x, GradientDirection::VERTICAL, &prewitt_kernel, false)).collect();

        RGBDOctave{gray_images,x_gradients,y_gradients,sigmas}

    }

}