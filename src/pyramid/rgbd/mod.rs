use crate::image::Image;
use self::{rgbd_octave::RGBDOctave, rgbd_runtime_parameters::RGBDRuntimeParameters};

pub mod rgbd_octave;
pub mod rgbd_runtime_parameters;

#[derive(Debug,Clone)]
pub struct RGBDPyramid<T> {
    pub octaves: Vec<T>,
    pub depth_image: Image
}
pub fn build_rgbd_pyramid(base_gray_image: Image, depth_image: Image, runtime_parameters: &RGBDRuntimeParameters) -> RGBDPyramid<RGBDOctave> {

    let mut octaves: Vec<RGBDOctave> = Vec::with_capacity(runtime_parameters.octave_count);

    let mut octave_image = base_gray_image;
    let mut sigma = runtime_parameters.sigma;

    for i in 0..runtime_parameters.octave_count {

        if i > 0 {
            octave_image = Image::downsample_half(&octaves[i-1].gray_images[0], false,  runtime_parameters.min_image_dimensions);
            sigma *= 2.0;
        }

        let new_octave = RGBDOctave::build_octave(&octave_image, sigma, runtime_parameters);

        octaves.push(new_octave);
    }

    RGBDPyramid {octaves,depth_image}
}