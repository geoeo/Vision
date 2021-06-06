extern crate image as image_rs;

use crate::image::Image;
use crate::filter::{prewitt_kernel::PrewittKernel,gauss_kernel::GaussKernel1D,gaussian_2_d_convolution,filter_1d_convolution};
use crate::pyramid::gd::gd_runtime_parameters::GDRuntimeParameters;
use crate::{Float,GradientDirection};


#[derive(Debug,Clone)]
pub struct GDOctave {
    pub gray_images: Vec<Image>,
    pub x_gradients: Vec<Image>,
    pub y_gradients: Vec<Image>,
    pub sigmas: Vec<Float>
}

impl GDOctave {

    pub fn build_octave(base_gray_image: &Image, sigma: Float, runtime_parameters: &GDRuntimeParameters) -> GDOctave {

        let mean = 0.0;
        let step = 1;

        let blur_radius = runtime_parameters.blur_radius;
        let prewitt_kernel = PrewittKernel::new();

        let sigmas = vec!(sigma);
        let kernels: Vec<GaussKernel1D> = sigmas.iter().map(|&sigma| GaussKernel1D::new(mean, sigma,step,blur_radius)).collect();
        let gray_images: Vec<Image> = match runtime_parameters.use_blur {
            false => vec!(base_gray_image.normalize()),
            _ => kernels.iter().map(|kernel| gaussian_2_d_convolution(&base_gray_image.normalize(), kernel, runtime_parameters.normalize_gray)).collect() //TODO make this a parameter
        };
        let images_borrows: Vec<&Image> = gray_images.iter().map(|x| x).collect();

        let mut x_gradients: Vec<Image> = (0..sigmas.len()).map(|x| filter_1d_convolution(&images_borrows,x, GradientDirection::HORIZINTAL, &prewitt_kernel, runtime_parameters.normalize_gradients)).collect();
        let mut y_gradients: Vec<Image> = (0..sigmas.len()).map(|x| filter_1d_convolution(&images_borrows,x, GradientDirection::VERTICAL, &prewitt_kernel, runtime_parameters.normalize_gradients)).collect();

        match runtime_parameters.invert_grad_x {
            true => x_gradients.iter_mut().for_each(|image| image.buffer *=-1.0),
            false => ()
        };

        //TODO: make bluring gradient cleaner/explixit ? 
        match runtime_parameters.blur_grad_x {
            true => {
                x_gradients = x_gradients.iter().enumerate().map(|(idx,image)|  gaussian_2_d_convolution(image, &kernels[idx], false)).collect::<Vec<Image>>();
                ()
            },
            false => ()
        };

        match runtime_parameters.invert_grad_y {
            true => y_gradients.iter_mut().for_each(|image| image.buffer *= -1.0),
            false => ()
        };

        match runtime_parameters.blur_grad_y {
            true => {
                y_gradients = y_gradients.iter().enumerate().map(|(idx,image)|  gaussian_2_d_convolution(image, &kernels[idx], false)).collect::<Vec<Image>>();
                ()
            },
            false => ()
        };




        GDOctave{gray_images,x_gradients,y_gradients,sigmas}

    }

}