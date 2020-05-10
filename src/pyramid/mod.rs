extern crate image as image_rs;

use image_rs::GrayImage;
use self::octave::Octave;

pub mod octave;

#[derive(Debug,Clone)]
pub struct Pyramid {
    pub octaves: Vec<Octave>,
    pub s: f32
}

impl Pyramid {
    //pub fn build_pyramid(base_image: &GrayImage, s: f32, octave_count: u8) -> Pyramid {
    pub fn build_pyramid(base_image: &GrayImage, s: f32, octave_count: u8) -> () {

        //TODO


    }
}