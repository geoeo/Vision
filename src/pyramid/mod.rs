extern crate image as image_rs;

use image_rs::GrayImage;
use self::octave::Octave;
use crate::image::{Image, filter, gauss_kernel::GaussKernel1D};
use crate::pyramid::runtime_params::RuntimeParams;

pub mod octave;
pub mod runtime_params;

#[derive(Debug,Clone)]
pub struct Pyramid {
    pub octaves: Vec<Octave>,
    pub s: usize //TODO: maybe save runtime params
}

impl Pyramid {
    pub fn build_pyramid(base_gray_image: &GrayImage, runtime_params: &RuntimeParams) -> Pyramid {
        let mut octaves: Vec<Octave> = Vec::with_capacity(runtime_params.octave_count);

        let base_image = Image::from_gray_image(base_gray_image, false);
        let upsample = Image::upsample_double(&base_image, false);

        //TODO: upscale and interpolate


        //TODO: check this
        let blur_width = Octave::generate_blur_half_width(runtime_params.blur_half_factor, runtime_params.sigma_in);
        //let blur_width = runtime_params.blur_half_factor;
        let kernel = GaussKernel1D::new(0.0, runtime_params.sigma_in,1,blur_width);
        //let initial_blur =  filter::gaussian_2_d_convolution(&base_image, &kernel, false);
        let initial_blur =  filter::gaussian_2_d_convolution(&upsample, &kernel, false);
        
        let mut octave_image = initial_blur;
        let mut sigma = runtime_params.sigma_initial;


        for i in 0..runtime_params.octave_count {

            if i > 0 {
                octave_image = Image::downsample_half(&octaves[i-1].images[runtime_params.sigma_count], false);
                sigma = octaves[i-1].sigmas[runtime_params.sigma_count];
            }

            let new_octave = Octave::build_octave(&octave_image, runtime_params.sigma_count, sigma, runtime_params);

            octaves.push(new_octave);
            

        }

    Pyramid{octaves,s: runtime_params.sigma_count}
    }


}