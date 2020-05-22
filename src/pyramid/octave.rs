extern crate image as image_rs;

use crate::image::{Image, filter, gauss_kernel::GaussKernel};
use crate::Float;

#[derive(Debug,Clone)]
pub struct Octave {
    pub images: Vec<Image>,
    pub difference_of_gaussians: Vec<Image>,
    pub sigmas: Vec<Float>
}

impl Octave {

    pub fn build_octave(base_image: &Image, s: usize, sigma_initial: Float) -> Octave {

        let image_count = s + 3;
        let range = 0..image_count;
        let mean = 0.0;
        let end = 3;
        let step = 1;

        let sigmas: Vec<Float> = range.map(|x| sigma_initial*Octave::generate_k(x as Float, s as Float)).collect();
        let kernels: Vec<GaussKernel> = sigmas.iter().map(|&sigma| GaussKernel::new(mean, sigma,step,end)).collect();
        let images: Vec<Image> = kernels.iter().map(|kernel| filter::gaussian_2_d_convolution(&base_image, kernel)).collect();

        let mut difference_of_gaussians: Vec<Image> = Vec::with_capacity(image_count-1);
        for i in 0..images.len()-1 {

            let difference_buffer = &images[i+1].buffer - &images[i].buffer;
            difference_of_gaussians.push(Image{buffer: difference_buffer, original_encoding: base_image.original_encoding});
        }

        Octave {images,difference_of_gaussians,sigmas}
    }

    pub fn index_for_next_octave(octave: &Octave) -> usize {
        octave.images.len()-3
    }

    fn generate_k(n: Float, s: Float) -> Float {
        assert!(n >= 0.0);
        let exp = n/s;
        exp.exp2()
    }
}

