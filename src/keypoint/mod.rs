extern crate nalgebra as na;

use crate::image::{Image,kernel::Kernel};
use crate::Float;
use crate::pyramid::octave::Octave;
use na::DMatrix;

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    x: usize,
    y: usize,
    sigma: Float
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

    let kernel = filter_kernel.kernel();
    let step = filter_kernel.step();
    let repeat = filter_kernel.half_repeat() as isize;
    let repeat_range = -repeat..repeat+1;
    let kernel_half_width = filter_kernel.half_width();
    let kernel_half_width_signed = kernel_half_width as isize;
    
    let source = &source_octave.images[0]; //TODO: select correct image
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();

    let x = 0; let y = 0; //TODO: make input params

    for kenel_idx in (-kernel_half_width_signed..kernel_half_width_signed+1).step_by(step){

        for repeat_idx in repeat_range.clone() {
            let sample_value = match gradient_direction {
                GradientDirection::HORIZINTAL => {
                    let sample_idx = (x as isize)+kenel_idx;
                    let y_repeat_idx =  match y as isize + repeat_idx {
                        y_idx if y_idx < 0 => 0,
                        y_idx if y_idx >= height as isize => height-1,
                        y_idx => y_idx as usize
                    };
                    
                    match sample_idx {
                        sample_idx if sample_idx < 0 =>  buffer.index((y_repeat_idx,0)),
                        sample_idx if sample_idx >=  (width-kernel_half_width) as isize => buffer.index((y_repeat_idx,width -1)),
                        _ => buffer.index((y_repeat_idx,sample_idx as usize))
                    }
                },
                GradientDirection::VERTICAL => {
                    let sample_idx = (y as isize)+kenel_idx;
                    let x_repeat_idx = match x as isize + repeat_idx {
                        x_idx if x_idx < 0 => 0,
                        x_idx if x_idx >= width as isize => width-1,
                        x_idx => x_idx as usize
                    };

                    match sample_idx {
                        sample_idx if sample_idx < 0 =>  buffer.index((0,x_repeat_idx)),
                        sample_idx if sample_idx >=  (height-kernel_half_width) as isize => buffer.index((height -1,x_repeat_idx)),
                        _ =>  buffer.index((sample_idx as usize, x_repeat_idx))
                    }
                },
                GradientDirection::SIGMA => { 
                    buffer.index((0, 0)) //TODO
                }
                
            };
            let kenel_value = kernel[(kenel_idx + kernel_half_width_signed) as usize];
        }   
    }


    ExtremaParameters {x: 0, y: 0, sigma: 0.0} //TODO

}