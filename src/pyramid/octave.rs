extern crate image as image_rs;

use image_rs::GrayImage;
use crate::image::Image;

#[derive(Debug,Clone)]
pub struct Octave {
    pub images: Vec<Image>,
    pub sigmas: Vec<f32>
}

impl Octave {

    pub fn build_octave(gray_image: &GrayImage, s: u32, sigma_initial: f32) -> Octave {

        let image_count = s + 3;
        let range = 0..image_count;
        let base_image = Image::from_gray_image(gray_image);

        let sigmas: Vec<f32> = range.map(|x| sigma_initial*Octave::generate_k(x as f32, s as f32)).collect();
        let images = sigmas.iter().map(|&sigma| Image::blur(&base_image, sigma)).collect();

        Octave {images,sigmas}
    }

    pub fn base_image_for_next_octave(octave: &Octave) -> &Image {
        let size = octave.images.len();
        &octave.images[size-3]
    }

    fn generate_k(n: f32, s: f32) -> f32 {
        assert!(n <= s && n >= 0f32);
        let exp = n/s;
        exp.exp2()
    }
}

