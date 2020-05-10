extern crate image as image_rs;

use image_rs::{GrayImage, imageops::blur};
use crate::image::Image;

#[derive(Debug,Clone)]
pub struct Octave {
    pub images: Vec<Image>,
    pub sigmas: Vec<f32>
}

impl Octave {

    pub fn add(octave: &mut Octave, image: Image, sigma: f32) -> () {
        octave.images.push(image);
        octave.sigmas.push(sigma);
    }

    pub fn blur(gray_image: &GrayImage, sigma: f32) -> Image {
        let blurred_gray_imaage = blur(gray_image,sigma);
        Image::from_gray_image(&blurred_gray_imaage)
    }

    pub fn blur_and_add(octave: &mut Octave, gray_image: &GrayImage, sigma: f32) -> () {
        let blurred_image = Octave::blur(gray_image, sigma);

        Octave::add(octave, blurred_image, sigma);
    }
}

