extern crate image as image_rs;

use image_rs::GrayImage;
use self::octave::Octave;
use crate::Float;
use crate::image::Image;

pub mod octave;

#[derive(Debug,Clone)]
pub struct Pyramid {
    pub octaves: Vec<Octave>,
    pub s: usize
}

impl Pyramid {
    pub fn build_pyramid(base_gray_image: &GrayImage, s: usize, octave_count: usize, sigma_initial: Float) -> Pyramid {
        let mut octaves: Vec<Octave> = Vec::with_capacity(octave_count);

        let base_image = Image::from_gray_image(base_gray_image);
        let mut octave_image = &base_image;
        for i in 0..octave_count {

            octaves[i] = Octave::build_octave(octave_image, s, sigma_initial);
            octave_image = Octave::base_image_for_next_octave(&octaves[i]);
        }

    Pyramid{octaves,s}
    }
}