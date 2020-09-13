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

        let base_image = Image::from_gray_image(base_gray_image, false);
        let mut octave_image = base_image;
        let mut sigma = sigma_initial;



        for i in 0..octave_count {

            if i > 0 {
                octave_image = Image::downsample_half(&octaves[i-1].images[s]);
                sigma = octaves[i-1].sigmas[s];
            }

            let new_octave = Octave::build_octave(&octave_image, s, sigma);

            octaves.push(new_octave);
            

        }

    Pyramid{octaves,s}
    }


}