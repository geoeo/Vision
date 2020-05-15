extern crate image as image_rs;

use image_rs::GrayImage;
use crate::image::{Image, filter};
use crate::Float;

#[derive(Debug,Clone)]
pub struct Octave {
    pub images: Vec<Image>,
    pub sigmas: Vec<Float>
}

impl Octave {

    pub fn build_octave(gray_image: &GrayImage, s: u32, sigma_initial: Float) -> Octave {

        let image_count = s + 3;
        let range = 0..image_count;
        let base_image = Image::from_gray_image(gray_image);

        let sigmas: Vec<Float> = range.map(|x| sigma_initial*Octave::generate_k(x as Float, s as Float)).collect();
        let images = sigmas.iter().map(|&sigma| filter::blur(&base_image, 0.0, sigma,1,3)).collect();

        Octave {images,sigmas}
    }

    pub fn base_image_for_next_octave(octave: &Octave) -> &Image {
        let size = octave.images.len();
        &octave.images[size-3]
    }

    fn generate_k(n: Float, s: Float) -> Float {
        assert!(n <= s && n >= 0.0);
        let exp = n/s;
        exp.exp2()
    }
}

