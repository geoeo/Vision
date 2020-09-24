extern crate image as image_rs;

use crate::image::{Image, filter, gauss_kernel::GaussKernel, prewitt_kernel::PrewittKernel};
use crate::{Float,GradientDirection};
use crate::pyramid::runtime_params::RuntimeParams;

#[derive(Debug,Clone)]
pub struct Octave {
    pub images: Vec<Image>,
    pub x_gradient: Vec<Image>,
    pub y_gradient: Vec<Image>,
    pub dog_x_gradient: Vec<Image>,
    pub dog_y_gradient: Vec<Image>,
    pub dog_s_gradient: Vec<Image>,
    pub difference_of_gaussians: Vec<Image>,
    pub sigmas: Vec<Float>
}

impl Octave {

    pub fn build_octave(base_image: &Image, s: usize, sigma_initial: Float, runtime_params: &RuntimeParams) -> Octave {

        let image_count = s + 3;
        let range = 0..image_count;
        let mean = 0.0;
        let half_width = runtime_params.blur_half_width;
        let step = 1;

        let gradient_kernel = PrewittKernel::new();


        let sigmas: Vec<Float> = range.map(|x| sigma_initial*Octave::generate_k(x as Float, s as Float)).collect();
        let kernels: Vec<GaussKernel> = sigmas.iter().map(|&sigma| GaussKernel::new(mean, sigma,step,half_width)).collect();
        let images: Vec<Image> = kernels.iter().map(|kernel| filter::gaussian_2_d_convolution(&base_image, kernel, false)).collect();
        let x_gradient = images.iter().map(|x| filter::filter_1d_convolution_no_sigma(x, GradientDirection::HORIZINTAL, &gradient_kernel, false)).collect();
        let y_gradient = images.iter().map(|x| filter::filter_1d_convolution_no_sigma(x, GradientDirection::VERTICAL, &gradient_kernel, false)).collect();



        let mut difference_of_gaussians: Vec<Image> = Vec::with_capacity(image_count-1);
        for i in 0..images.len()-1 {

            let difference_buffer = &images[i+1].buffer - &images[i].buffer;
            difference_of_gaussians.push(Image::from_matrix(&difference_buffer, base_image.original_encoding, true));
        }
        
        let dog_range = 0..difference_of_gaussians.len();

        let dog_x_gradient = dog_range.clone().map(|sigma_idx| filter::filter_1d_convolution(&difference_of_gaussians,sigma_idx, GradientDirection::HORIZINTAL, &gradient_kernel, true)).collect();
        let dog_y_gradient = dog_range.clone().map(|sigma_idx| filter::filter_1d_convolution(&difference_of_gaussians,sigma_idx, GradientDirection::VERTICAL, &gradient_kernel,true )).collect();
        let dog_s_gradient = dog_range.clone().map(|sigma_idx| filter::filter_1d_convolution(&difference_of_gaussians,sigma_idx, GradientDirection::SIGMA, &gradient_kernel, true)).collect();
        Octave {images,x_gradient,y_gradient,dog_x_gradient,dog_y_gradient,dog_s_gradient,difference_of_gaussians,sigmas}
    }

    fn generate_k(n: Float, s: Float) -> Float {
        assert!(n >= 0.0);
        let exp = n/s;
        exp.exp2()
    }
}

