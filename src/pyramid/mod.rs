extern crate image as image_rs;

use image_rs::GrayImage;
use sift_octave::SiftOctave;
use crate::image::Image;
use crate::filter::{gauss_kernel::GaussKernel1D,gaussian_2_d_convolution};
use crate::pyramid::sift_runtime_params::SiftRuntimeParams;

pub mod sift_octave;
pub mod sift_runtime_params;

#[derive(Debug,Clone)]
pub struct Pyramid<T> {
    pub octaves: Vec<T>,
    pub sigma_count: usize
}


pub fn build_sift_pyramid(base_gray_image: &GrayImage, runtime_params: &SiftRuntimeParams) -> Pyramid<SiftOctave> {
    let mut octaves: Vec<SiftOctave> = Vec::with_capacity(runtime_params.octave_count);

    let base_image = Image::from_gray_image(base_gray_image, false);
    let upsample = Image::upsample_double(&base_image, false);

    //TODO: check this
    let blur_width = SiftOctave::generate_blur_half_width(runtime_params.blur_half_factor, runtime_params.sigma_in);
    let kernel = GaussKernel1D::new(0.0, runtime_params.sigma_in,1,blur_width);
    let initial_blur =  gaussian_2_d_convolution(&upsample, &kernel, false);
    
    let mut octave_image = initial_blur;
    let mut sigma = runtime_params.sigma_initial;
    let sigma_count = runtime_params.sigma_count;


    for i in 0..runtime_params.octave_count {

        if i > 0 {
            octave_image = Image::downsample_half(&octaves[i-1].images[runtime_params.sigma_count], false);
            sigma = octaves[i-1].sigmas[sigma_count];
        }

        let new_octave = SiftOctave::build_octave(&octave_image, runtime_params.sigma_count, sigma, runtime_params);

        octaves.push(new_octave);
        

    }

    Pyramid{octaves,sigma_count}
}


