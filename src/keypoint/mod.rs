extern crate nalgebra as na;

use crate::image::{Image,kernel::Kernel,prewitt_kernel::PrewittKernel, laplace_kernel::LaplaceKernel};
use crate::Float;
use crate::pyramid::octave::Octave;
use na::DMatrix;

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize
} 

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum GradientDirection {
    HORIZINTAL,
    VERTICAL,
    SIGMA
}

//TODO: encapsulate input params better
pub fn detect_extrema(source_octave: &Octave, sigma_level: usize,x_offset: usize, y_offset: usize , x_step: usize, y_step: usize) -> Vec<ExtremaParameters> {

    let mut extrema_vec: Vec<ExtremaParameters> = Vec::new();

    let image_buffer = &source_octave.difference_of_gaussians[sigma_level].buffer;
    let prev_buffer = &source_octave.difference_of_gaussians[sigma_level-1].buffer;
    let next_buffer = &source_octave.difference_of_gaussians[sigma_level+1].buffer;

    for x in (x_offset..image_buffer.ncols()-x_offset).step_by(x_step) {
        for y in (y_offset..image_buffer.nrows()-y_offset).step_by(y_step)  {

            let sample_value = image_buffer[(y,x)];

            //TODO: @Investigate parallel
            let is_extrema = 
                is_sample_extrema_in_neighbourhood(sample_value,x,y,image_buffer,true) && 
                is_sample_extrema_in_neighbourhood(sample_value,x,y,prev_buffer,false) && 
                is_sample_extrema_in_neighbourhood(sample_value,x,y,next_buffer,false);

            if is_extrema {
                extrema_vec.push(ExtremaParameters{x,y,sigma_level});
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

pub fn extrema_refinement(extrema: &Vec<ExtremaParameters>, source_octave: &Octave) -> Vec<ExtremaParameters> {

    extrema.iter().cloned().filter(|x| contrast_rejection(source_octave, x)).collect()

}

pub fn contrast_rejection(source_octave: &Octave, input_params: &ExtremaParameters) -> bool {


    let first_order_derivative_filter = PrewittKernel::new(1);
    let second_order_derivative_filter = LaplaceKernel::new(1);

    let first_order_derivative_x = gradient_eval(source_octave,input_params,&first_order_derivative_filter,GradientDirection::HORIZINTAL);
    let first_order_derivative_y = gradient_eval(source_octave,input_params,&first_order_derivative_filter,GradientDirection::VERTICAL);
    let first_order_derivative_sigma = gradient_eval(source_octave,input_params,&first_order_derivative_filter,GradientDirection::SIGMA);

    let second_order_derivative_x = gradient_eval(source_octave,input_params,&second_order_derivative_filter,GradientDirection::HORIZINTAL);
    let second_order_derivative_y = gradient_eval(source_octave,input_params,&second_order_derivative_filter,GradientDirection::VERTICAL);
    let second_order_derivative_sigma = gradient_eval(source_octave,input_params,&second_order_derivative_filter,GradientDirection::SIGMA);

    let pertub_x = first_order_derivative_x/second_order_derivative_x;
    let pertub_y = first_order_derivative_y/second_order_derivative_y;
    let pertub_sigma = first_order_derivative_sigma/second_order_derivative_sigma;

    if pertub_x > 0.5 || pertub_y > 0.5 || pertub_sigma > 0.5 {
        return false;
    }

    let dog_x = source_octave.difference_of_gaussians[input_params.sigma_level].buffer.index((input_params.y,input_params.x));
    let first_order_derivative_pertub_dot = first_order_derivative_x*pertub_x + first_order_derivative_y*pertub_y+first_order_derivative_sigma+pertub_sigma;
    let dog_x_pertub = dog_x + 0.5*first_order_derivative_pertub_dot;

    //TODO: make sure difference of gaussians are normalized to 0..1 range
    dog_x_pertub.abs() >= 0.3

}

fn gradient_eval(source_octave: &Octave,input_params: &ExtremaParameters, filter_kernel: &dyn Kernel, gradient_direction: GradientDirection) -> Float {

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

    let width = source_octave.difference_of_gaussians[sigma_level_input].buffer.ncols();
    let height = source_octave.difference_of_gaussians[sigma_level_input].buffer.nrows();

    match gradient_direction {
        GradientDirection::HORIZINTAL => {
            assert!(x_input_signed -kernel_half_width_signed >= 0 && x_input + kernel_half_width <= width);
            assert!(x_input_signed -repeat >= 0 && x_input + (repeat as usize) < width);
         },
        GradientDirection::VERTICAL => { 
            assert!(y_input_signed -kernel_half_width_signed >= 0 && y_input + kernel_half_width <= height);
            assert!(y_input_signed -repeat >= 0 && y_input + (repeat as usize) < height);
         },
        GradientDirection::SIGMA => { 
            assert!(sigma_level_input_signed -kernel_half_width_signed >= 0 && sigma_level_input + kernel_half_width < source_octave.difference_of_gaussians.len());
        }

    }

    let mut convolved_value = 0.0;
    for kenel_idx in (-kernel_half_width_signed..kernel_half_width_signed+1).step_by(step){
        let kenel_value = kernel[(kenel_idx + kernel_half_width_signed) as usize];

            let weighted_sample = match gradient_direction {
                GradientDirection::HORIZINTAL => {
                    let mut acc = 0.0;
                    let source = &source_octave.difference_of_gaussians[sigma_level_input]; 
                    let buffer = &source.buffer;
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
                    let source = &source_octave.difference_of_gaussians[sigma_level_input]; 
                    let buffer = &source.buffer;
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
                    let sample_buffer =  &source_octave.difference_of_gaussians[sample_idx as usize].buffer;
                    let sample_value = sample_buffer.index((y_input,x_input));
                    sample_value*kenel_value
                }
                
            };

            convolved_value += weighted_sample;
    
    }

    convolved_value
}