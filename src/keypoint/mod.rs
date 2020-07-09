extern crate nalgebra as na;

use crate::image::{kernel::Kernel,filter::gradient_convolution_at_sample};
use crate::{Float,ExtremaParameters, GradientDirection};
use crate::pyramid::octave::Octave;
use na::DMatrix;

mod hessian;


pub fn detect_extrema(source_octave: &Octave, sigma_level: usize, filter_half_width: usize,filter_half_repeat: usize, x_step: usize, y_step: usize) -> Vec<ExtremaParameters> {

    let mut extrema_vec: Vec<ExtremaParameters> = Vec::new();

    let image_buffer = &source_octave.difference_of_gaussians[sigma_level].buffer;
    let prev_buffer = &source_octave.difference_of_gaussians[sigma_level-1].buffer;
    let next_buffer = &source_octave.difference_of_gaussians[sigma_level+1].buffer;

    let gradient_range = 1;
    let offset = std::cmp::max(filter_half_width,filter_half_repeat)+gradient_range;

    for x in (offset..image_buffer.ncols()-offset).step_by(x_step) {
        for y in (offset..image_buffer.nrows()-offset).step_by(y_step)  {

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

pub fn extrema_refinement(extrema: &Vec<ExtremaParameters>, source_octave: &Octave, first_order_kernel: &dyn Kernel, second_order_kernel: &dyn Kernel) -> Vec<ExtremaParameters> {

    assert!(second_order_kernel.half_repeat() <= first_order_kernel.half_repeat());
    assert!(second_order_kernel.half_width() <= first_order_kernel.half_width());

    extrema.iter().cloned().filter(|x| contrast_rejection(source_octave, x, first_order_kernel,second_order_kernel)).filter(|x| edge_response_rejection(source_octave, x,second_order_kernel,10)).collect()
}

pub fn contrast_rejection(source_octave: &Octave, input_params: &ExtremaParameters, first_order_kernel: &dyn Kernel, second_order_kernel: &dyn Kernel) -> bool {

    let first_order_derivative_x = gradient_convolution_at_sample(source_octave,input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let first_order_derivative_y = gradient_convolution_at_sample(source_octave,input_params,first_order_kernel,GradientDirection::VERTICAL);
    let first_order_derivative_sigma = gradient_convolution_at_sample(source_octave,input_params,first_order_kernel,GradientDirection::SIGMA);

    let second_order_derivative_x = gradient_convolution_at_sample(source_octave,input_params,second_order_kernel,GradientDirection::HORIZINTAL);
    let second_order_derivative_y = gradient_convolution_at_sample(source_octave,input_params,second_order_kernel,GradientDirection::VERTICAL);
    let second_order_derivative_sigma = gradient_convolution_at_sample(source_octave,input_params,second_order_kernel,GradientDirection::SIGMA);

    let pertub_x = first_order_derivative_x/second_order_derivative_x;
    let pertub_y = first_order_derivative_y/second_order_derivative_y;
    let pertub_sigma = first_order_derivative_sigma/second_order_derivative_sigma;

    if pertub_x > 0.5 || pertub_y > 0.5 || pertub_sigma > 0.5 {
        return false;
    }

    let dog_x = source_octave.difference_of_gaussians[input_params.sigma_level].buffer.index((input_params.y,input_params.x));
    let first_order_derivative_pertub_dot = first_order_derivative_x*pertub_x + first_order_derivative_y*pertub_y+first_order_derivative_sigma+pertub_sigma;
    let dog_x_pertub = dog_x + 0.5*first_order_derivative_pertub_dot;

    dog_x_pertub.abs() >= 0.03

}

pub fn edge_response_rejection(source_octave: &Octave, input_params: &ExtremaParameters, second_order_kernel: &dyn Kernel, r: usize) -> bool {
    let hessian = hessian::new(source_octave,input_params,second_order_kernel);
    hessian::eval_hessian(&hessian, r)
}

