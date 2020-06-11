extern crate nalgebra as na;

use crate::image::{Image,kernel::Kernel};
use crate::Float;
use crate::pyramid::octave::Octave;
use na::DMatrix;

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    x: usize,
    y: usize,
    sigma_level: usize
} 

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum GradientDirection {
    HORIZINTAL,
    VERTICAL,
    SIGMA
}

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

//TODO: Refactor the method for single (x,y,sigma) coordinate / Maybe make a new method
pub fn keypoint_localization(source_octave: &Octave, input_params: ExtremaParameters, gradient_direction: GradientDirection, filter_kernel: &dyn Kernel) -> ExtremaParameters {

    let x_input = input_params.x; 
    let x_input_signed = input_params.x as isize; 
    let y_input = input_params.y; 
    let y_input_signed = input_params.y as isize; 
    let sigma_level_input = input_params.sigma_level;
    let sigma_level_input_signed = input_params.sigma_level as isize;

    let kernel = filter_kernel.kernel();
    let step = filter_kernel.step();
    let repeat = filter_kernel.half_repeat() as isize;
    let repeat_range = -repeat..repeat+1;
    let kernel_half_width = filter_kernel.half_width();
    let kernel_half_width_signed = kernel_half_width as isize;
    
    let source = &source_octave.images[sigma_level_input]; 
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();


    assert!(x_input_signed -kernel_half_width_signed >= 0 && x_input + kernel_half_width <= width);
    assert!(x_input_signed -repeat >= 0 && x_input + ((repeat+1) as usize) < width);

    assert!(y_input_signed -kernel_half_width_signed >= 0 && y_input + kernel_half_width <= height);
    assert!(y_input_signed -repeat >= 0 && y_input + ((repeat+1) as usize) < height);

    assert!(sigma_level_input_signed -kernel_half_width_signed > 0 && sigma_level_input + kernel_half_width < source_octave.sigmas.len());
    assert!(sigma_level_input_signed - repeat > 0 && sigma_level_input + ((repeat+1) as usize) < source_octave.sigmas.len());
    
    
    //TODO: Refactor so that the following GradientDirection calls are encapsulated
    let mut convolved_value = 0.0;
    for kenel_idx in (-kernel_half_width_signed..kernel_half_width_signed+1).step_by(step){
        let kenel_value = kernel[(kenel_idx + kernel_half_width_signed) as usize];

            let weighted_sample = match gradient_direction {
                GradientDirection::HORIZINTAL => {
                    let mut acc = 0.0;
                    for repeat_idx in repeat_range.clone() {
                        let sample_idx = x_input_signed +kenel_idx;
                        let y_repeat_idx =  y_input_signed + repeat_idx;

                        let sample_value = buffer.index((y_repeat_idx as usize,sample_idx as usize));
                        acc += kenel_value*sample_value;
                    }
                    let range_size = repeat_range.end - repeat_range.start;
                    acc/ range_size as Float
                },
                GradientDirection::VERTICAL => {
                    let mut acc = 0.0;
                    for repeat_idx in repeat_range.clone() {
                        let sample_idx = y_input_signed+kenel_idx;
                        let x_repeat_idx = x_input_signed + repeat_idx;
                        let sample_value = buffer.index((sample_idx as usize, x_repeat_idx as usize));
                        acc += kenel_value*sample_value;
                    }
                    let range_size = repeat_range.end - repeat_range.start;
                    acc/ range_size as Float
                },
                GradientDirection::SIGMA => { 
                    //TODO: Not sure how repeat/2D kernels should work along the sigma axis
                    let sample_idx = sigma_level_input_signed + kenel_idx;
                    let sample_buffer =  &source_octave.images[sample_idx as usize].buffer;
                    let sample_value = sample_buffer.index((y_input,x_input));
                    sample_value*kenel_value
                }
                
            };

            convolved_value += weighted_sample;
    
    }


    ExtremaParameters {x: 0, y: 0, sigma_level: 0} //TODO

}