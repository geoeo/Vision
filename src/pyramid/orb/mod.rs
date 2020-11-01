extern crate image as image_rs;

use image_rs::GrayImage;

use crate::image::Image;
use crate::pyramid::Pyramid;
use self::{orb_octave::OrbOctave, orb_runtime_parameters::OrbRuntimeParameters};


pub mod orb_octave;
pub mod orb_runtime_parameters;

pub fn build_orb_pyramid(base_gray_image: &GrayImage, runtime_parameters: &OrbRuntimeParameters) -> Pyramid<OrbOctave> {

    let mut octaves: Vec<OrbOctave> = Vec::with_capacity(runtime_parameters.octave_count);
    let base_image = Image::from_gray_image(base_gray_image, false);

    let mut octave_image = base_image;
    let mut sigma = runtime_parameters.sigma;

    for i in 0..runtime_parameters.octave_count {

        if i > 0 {
            octave_image = Image::downsample_half(&octaves[i-1].images[0], false);
            sigma *= 2.0;
        }

        let new_octave = OrbOctave::build_octave(&octave_image,sigma, runtime_parameters);

        octaves.push(new_octave);
    }

    Pyramid {octaves}
}
