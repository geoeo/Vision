extern crate nalgebra as na;

use na::{Matrix3x1,Matrix3,LU};
use crate::image::{kernel::Kernel,filter::gradient_convolution_at_sample};
use crate::{Float,ExtremaParameters, GradientDirection};
use crate::pyramid::octave::Octave;
use na::DMatrix;

mod hessian;


pub fn detect_extrema(source_octave: &Octave, sigma_level: usize, filter_half_width: usize,filter_half_repeat: usize, x_step: usize, y_step: usize) -> Vec<ExtremaParameters> {

    let mut extrema_vec: Vec<ExtremaParameters> = Vec::new();

    assert!(sigma_level+1 < source_octave.difference_of_gaussians.len());
    assert!(sigma_level > 0);

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

pub fn extrema_refinement(extrema: &Vec<ExtremaParameters>, source_octave: &Octave, first_order_kernel: &dyn Kernel) -> Vec<ExtremaParameters> {
    extrema.iter().cloned().filter(|x| contrast_filter(source_octave, x, first_order_kernel)).filter(|x| edge_response_filter(source_octave, x,first_order_kernel, 10)).collect()
}

pub fn contrast_filter(source_octave: &Octave, input_params: &ExtremaParameters, first_order_kernel: &dyn Kernel) -> bool {

    let dx = source_octave.dog_x_gradient[input_params.sigma_level].buffer[(input_params.y,input_params.x)];
    let dy = source_octave.dog_y_gradient[input_params.sigma_level].buffer[(input_params.y,input_params.x)];
    let ds = source_octave.dog_s_gradient[input_params.sigma_level].buffer[(input_params.y,input_params.x)];

    let b = Matrix3x1::new(dx,dy,ds);
    let perturb = interpolate(source_octave,input_params,first_order_kernel);
    let width = source_octave.dog_x_gradient[0].buffer.ncols() as isize;
    let height = source_octave.dog_x_gradient[0].buffer.nrows() as isize;
    let kernel_half_width = first_order_kernel.half_width() as isize;

    let input_x = input_params.x as isize;
    let input_y = input_params.y as isize;
    let input_s = input_params.sigma_level as isize;

    let (perturb_final,extrema_final) = match perturb {
        perturb if perturb[(0,0)] > 0.5 && input_x + 1 + kernel_half_width < width => {
            let extrema= ExtremaParameters{x:input_params.x + 1,y:input_params.y,sigma_level:input_params.sigma_level};
            (interpolate(source_octave,&extrema,first_order_kernel),extrema)
        },
        perturb if perturb[(0,0)] < -0.5  && input_x - 1  -kernel_half_width > 0 => {
            let extrema = ExtremaParameters{x:input_params.x -1,y:input_params.y,sigma_level:input_params.sigma_level};
            (interpolate(source_octave,&extrema,first_order_kernel),extrema)
        },
        perturb if perturb[(1,0)] > 0.5  && input_y + 1 + kernel_half_width < height => {
            let extrema = ExtremaParameters{x:input_params.x,y:input_params.y + 1,sigma_level:input_params.sigma_level};
            (interpolate(source_octave,&extrema,first_order_kernel),extrema)
        },
        perturb if perturb[(1,0)] < -0.5 && input_y - 1 - kernel_half_width > 0 => {
            let extrema = ExtremaParameters{x:input_params.x,y:input_params.y - 1,sigma_level:input_params.sigma_level};
            (interpolate(source_octave,&extrema,first_order_kernel),extrema)
        },
        perturb if perturb[(2,0)] > 0.5 && input_s + 1 + kernel_half_width  < source_octave.dog_s_gradient.len() as isize=> {
            let extrema = ExtremaParameters{x:input_params.x ,y:input_params.y,sigma_level:input_params.sigma_level + 1};
            (interpolate(source_octave,&extrema,first_order_kernel),extrema)
        },
        perturb if perturb[(2,0)] < -0.5 && input_s - 1 - kernel_half_width > 0 => {
            let extrema = ExtremaParameters{x:input_params.x + 1,y:input_params.y,sigma_level:input_params.sigma_level - 1};
            (interpolate(source_octave,&extrema,first_order_kernel),extrema)
        },
        _ => (perturb,input_params.clone())
    };


    let dog_sample = source_octave.difference_of_gaussians[extrema_final.sigma_level].buffer.index((extrema_final.y,extrema_final.x));
    let dog_x_pertub = dog_sample + 0.5*b.dot(&perturb_final);
    
    dog_x_pertub.abs() >= 0.03

}

pub fn edge_response_filter(source_octave: &Octave, input_params: &ExtremaParameters, first_order_kernel: &dyn Kernel, r: usize) -> bool {
    let hessian = hessian::new(source_octave,input_params,first_order_kernel);
    hessian::accept_hessian(&hessian, r)
}

fn interpolate(source_octave: &Octave, input_params: &ExtremaParameters, first_order_kernel: &dyn Kernel) -> Matrix3x1<Float> {

    let dx = source_octave.dog_x_gradient[input_params.sigma_level].buffer[(input_params.y,input_params.x)];
    let dy = source_octave.dog_y_gradient[input_params.sigma_level].buffer[(input_params.y,input_params.x)];
    let ds = source_octave.dog_s_gradient[input_params.sigma_level].buffer[(input_params.y,input_params.x)];

    let dxx = gradient_convolution_at_sample(source_octave,&source_octave.dog_x_gradient,input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample(source_octave,&source_octave.dog_y_gradient,input_params,first_order_kernel,GradientDirection::VERTICAL);
    let dss = gradient_convolution_at_sample(source_octave,&source_octave.dog_s_gradient,input_params,first_order_kernel,GradientDirection::SIGMA);

    let dxy = gradient_convolution_at_sample(source_octave,&source_octave.dog_x_gradient,input_params,first_order_kernel,GradientDirection::VERTICAL);
    let dxs = gradient_convolution_at_sample(source_octave,&source_octave.dog_x_gradient,input_params,first_order_kernel,GradientDirection::SIGMA);

    let dyx = gradient_convolution_at_sample(source_octave,&source_octave.dog_y_gradient,input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let dys = gradient_convolution_at_sample(source_octave,&source_octave.dog_y_gradient,input_params,first_order_kernel,GradientDirection::SIGMA);

    let dsx = gradient_convolution_at_sample(source_octave,&source_octave.dog_s_gradient,input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let dsy = gradient_convolution_at_sample(source_octave,&source_octave.dog_s_gradient,input_params,first_order_kernel,GradientDirection::VERTICAL);

    let a = Matrix3::new(dxx,dxy,dxs,
                         dyx,dyy,dys,
                         dsx,dsy,dss);

    let b = Matrix3x1::new(dx,dy,ds);
    a.lu().solve(&b).expect("Linear resolution failed.")
}

