extern crate image as image_rs;

use image_rs::GrayImage;


use crate::pyramid::Pyramid;
use self::{orb_octave::OrbOctave, orb_runtime_parameters::OrbRuntimeParameters};


pub mod orb_octave;
pub mod orb_runtime_parameters;

pub fn build_sift_pyramid(base_gray_image: &GrayImage, runtime_parameters: &OrbRuntimeParameters) -> Pyramid<OrbOctave> {

    panic!("not yet implemented");
}
