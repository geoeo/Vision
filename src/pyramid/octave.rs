extern crate image as image_rs;

use std::ops::Range;
use image_rs::{GrayImage, imageops::blur};
use crate::image::Image;

#[derive(Debug,Clone)]
pub struct Octave {
    pub images: Vec<Image>,
    pub sigmas: Vec<f32>
}

impl Octave {

    pub fn build_octave(base_image: &GrayImage, s: u32, sigma_initial: f32) -> Octave {

        let image_count = s + 3;
        let range = 0..image_count;
        let sigmas: Vec<f32> = range.map(|x| sigma_initial*Octave::generate_k(x as f32, s as f32)).collect();

        let images = sigmas.iter().map(|&sigma| Octave::blur(base_image, sigma)).collect();

        Octave {images,sigmas}
    }

    fn add(octave: &mut Octave, image: Image, sigma: f32) -> () {
        octave.images.push(image);
        octave.sigmas.push(sigma);
    }

    fn blur(gray_image: &GrayImage, sigma: f32) -> Image {
        let blurred_gray_image = blur(gray_image,sigma);
        Image::from_gray_image(&blurred_gray_image)
    }

    fn blur_and_add(octave: &mut Octave, gray_image: &GrayImage, sigma: f32) -> () {
        let blurred_image = Octave::blur(gray_image, sigma);

        Octave::add(octave, blurred_image, sigma);
    }

    fn generate_k(n: f32, s: f32) -> f32 {
        assert!(n <= s && n >= 0f32);
        let exp = n/s;
        exp.exp2()
    }
}

