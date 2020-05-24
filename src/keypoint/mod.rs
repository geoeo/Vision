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

            //TODO: @Investigate parallel
            let is_extrema = 
                is_sample_extrema_in_neighbourhood(sample_value,x,y,image_buffer,true) && 
                is_sample_extrema_in_neighbourhood(sample_value,x,y,prev_buffer,false) && 
                is_sample_extrema_in_neighbourhood(sample_value,x,y,next_buffer,false);

            if is_extrema {
                extrema_vec.push((x,y));
            }
        }
    }

    extrema_vec
}

fn is_sample_extrema_in_neighbourhood(sample: Float, x_sample: usize, y_sample: usize, neighbourhood_buffer: &DMatrix<Float>, skip_center: bool) -> bool {

    let mut is_smallest = true;
    let mut is_largest = true;

    for x in x_sample-1..x_sample+2 {
        for y in y_sample-1..y_sample+2 {

            if x == x_sample && y == y_sample && skip_center {
                continue;
            }

            let value = neighbourhood_buffer[(y,x)];
            is_smallest &= sample < value;
            is_largest &= sample > value;

            if !(is_smallest || is_largest) {
                break;
            }

        }
    }

    is_smallest || is_largest
}