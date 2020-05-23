extern crate nalgebra as na;

use crate::image::Image;
use crate::Float;
use na::DMatrix;

pub fn detect_extrema(image: &Image, prev_image: &Image, next_image: &Image, x_step: usize, y_step: usize) -> Vec<(usize,usize)> {

    let mut extrema_vec: Vec<(usize,usize)> = Vec::new();

    let image_buffer = &image.buffer;
    let prev_buffer = &prev_image.buffer;
    let next_buffer = &next_image.buffer;

    for x in (1..image_buffer.ncols()-1).step_by(x_step) {
        for y in (1..image_buffer.nrows()-1).step_by(y_step)  {

            let sample_value = image_buffer[(y,x)];

            //TODO: compare sample to 8 neighbours in image and 9 in prev and next
            
        }
    }

    extrema_vec
}

fn is_sample_extrema_in_neighbourhood(sample: Float, x: usize, y: usize, neighbourhood_buffer: &DMatrix<Float>, skip_center: bool) -> bool {
    false
}