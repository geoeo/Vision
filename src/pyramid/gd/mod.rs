use crate::image::Image;
use self::{gd_octave::GDOctave, gd_runtime_parameters::GDRuntimeParameters};

pub mod gd_octave;
pub mod gd_runtime_parameters;

#[derive(Debug,Clone)]
pub struct GDPyramid<T> {
    pub octaves: Vec<T>,
    pub depth_image: Image
}
pub fn build_rgbd_pyramid(base_gray_image: Image, depth_image: Image, runtime_parameters: &GDRuntimeParameters) -> GDPyramid<GDOctave> {

    let gray_image_shape = base_gray_image.buffer.shape();
    let depth_image_shape = depth_image.buffer.shape();
    assert_eq!(gray_image_shape,depth_image_shape);

    let mut octaves: Vec<GDOctave> = Vec::with_capacity(runtime_parameters.octave_count);

    let mut octave_image = base_gray_image.normalize();
    let mut sigma = runtime_parameters.sigma;

    for i in 0..runtime_parameters.octave_count {

        if i > 0 {
            octave_image = Image::downsample_half(&octaves[i-1].gray_images[0], false, runtime_parameters.pyramid_scale , runtime_parameters.min_image_dimensions);
            sigma *= 2.0;
        }

        let new_octave = GDOctave::build_octave(&octave_image, sigma, runtime_parameters);

        octaves.push(new_octave);
    }

    GDPyramid {octaves,depth_image}
}