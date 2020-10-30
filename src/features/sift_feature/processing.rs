extern crate nalgebra as na;

use na::{Matrix2,Matrix3x1, Matrix3};
use crate::pyramid::sift_octave::SiftOctave;
use crate::{Float, GradientDirection};
use crate::filter::{kernel::Kernel,gradient_convolution_at_sample_3_d,prewitt_kernel::PrewittKernel,laplace_kernel::LaplaceKernel,laplace_off_center_kernel::LaplaceOffCenterKernel};
use crate::features::{Feature,sift_feature::SiftFeature};


//TODO: @Investigate: maybe precomputing the gradient images is more efficient
pub fn harris_corner_matrix(source_octave: &SiftOctave, input_params: &dyn Feature) -> Matrix2<Float> {

    let second_order_kernel = LaplaceKernel::new();
    let first_order_kernel = PrewittKernel::new();


    let dxx = gradient_convolution_at_sample_3_d(&source_octave.difference_of_gaussians.iter().map(|x| x).collect(),input_params,&second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample_3_d(&source_octave.difference_of_gaussians.iter().map(|x| x).collect(),input_params,&second_order_kernel,GradientDirection::VERTICAL);
    let dxy = gradient_convolution_at_sample_3_d(&source_octave.dog_x_gradient.iter().map(|x| x).collect(),input_params,&first_order_kernel,GradientDirection::VERTICAL);

    Matrix2::new(dxx,dxy,
        dxy,dyy)

}

pub fn reject_edge(hessian: &Matrix2<Float>, r: Float) -> bool {
    let trace = hessian.trace();
    let determinant = hessian.determinant();
    let hessian_factor = trace.powi(2)/determinant;
    let r_factor = (r+1.0).powi(2)/r;

    hessian_factor < r_factor as Float && determinant > 0.0
}

pub fn accept_edge(hessian: &Matrix2<Float>, r: Float) -> bool {
    let trace = hessian.trace();
    let determinant = hessian.determinant();
    let hessian_factor = trace.powi(2)/determinant;
    let r_factor = (r+1.0).powi(2)/r;

    hessian_factor >= r_factor as Float && determinant > 0.0
}

//TODO: maybe return new extrema instead due to potential change of coordiantes in interpolation
//TODO: needs to be more stable
pub fn subpixel_refinement(source_octave: &SiftOctave, octave_level: usize, input_params:  &SiftFeature) -> (Float,SiftFeature) {

    let sigma_level = input_params.get_closest_sigma_level();

    let first_order_kernel = PrewittKernel::new();
    let second_order_kernel = LaplaceKernel::new();
    let second_order_off_center_kernel = LaplaceOffCenterKernel::new();


    let kernel_half_width = first_order_kernel.half_width();

    let s = source_octave.s();
    let sigma_range = (1.0/(s as Float)).exp2();
    let mut perturb_final = interpolate(source_octave,input_params,&first_order_kernel,&second_order_kernel);
    let mut extrema_final =  SiftFeature{x:input_params.x ,y:input_params.y,sigma_level:input_params.sigma_level};
    let max_it = 5; // TODO: put this in runtime params  
    let mut counter = 0;
    let cutoff = 0.6;

    perturb_final[(0,0)] = match  perturb_final[(0,0)] {
        v if v >= cutoff => 0.5,
        v if v <= -cutoff => -0.5,
        v => v
    };

    perturb_final[(1,0)] = match  perturb_final[(1,0)] {
        v if v >= cutoff => 0.5,
        v if v <= -cutoff => -0.5,
        v => v
    };

    perturb_final[(2,0)] = match  perturb_final[(2,0)] {
        v if v >= sigma_range/2.0 => sigma_range/2.0,
        v if v <= -sigma_range/2.0 => -sigma_range/2.0,
        v => v
    };


    extrema_final.x += perturb_final[(0,0)];
    extrema_final.y += perturb_final[(1,0)];
    extrema_final.sigma_level += perturb_final[(2,0)];


    //TODO: this seems buggy
    while (perturb_final[(0,0)].abs() > cutoff || perturb_final[(1,0)].abs() > cutoff || perturb_final[(2,0)].abs() > sigma_range/2.0) && counter < max_it  {
        
        // if perturb_final[(0,0)].abs() > cutoff {
        //     extrema_final.x += perturb_final[(0,0)];
        // }

        // if perturb_final[(1,0)].abs() > cutoff {
        //     extrema_final.y += perturb_final[(1,0)];
        // }
        
        // if  perturb_final[(2,0)].abs() > sigma_range/2.0 {
        //     extrema_final.sigma_level += perturb_final[(2,0)];
        // }

        perturb_final[(0,0)] = match  perturb_final[(0,0)] {
            v if v >= cutoff => 0.5,
            v if v <= -cutoff => -0.5,
            v => v
        };

        perturb_final[(1,0)] = match  perturb_final[(1,0)] {
            v if v >= cutoff => 0.5,
            v if v <= -cutoff => -0.5,
            v => v
        };

        perturb_final[(2,0)] = match  perturb_final[(2,0)] {
            v if v >= sigma_range/2.0 =>sigma_range/2.0,
            v if v <= -sigma_range/2.0 => -sigma_range/2.0,
            v => v
        };
    
        extrema_final.x += perturb_final[(0,0)];
        extrema_final.y += perturb_final[(1,0)];
        extrema_final.sigma_level += perturb_final[(2,0)];

        let closest_sigma_level = extrema_final.get_closest_sigma_level();

        if source_octave.within_range(extrema_final.get_x_image(), extrema_final.get_y_image(), closest_sigma_level, kernel_half_width) {
            perturb_final = interpolate(source_octave,&extrema_final, &first_order_kernel,&second_order_kernel);
            counter = counter +1;

        } else {
            counter = max_it;
        }

    }


    let closest_sigma_level = extrema_final.get_closest_sigma_level();
    if source_octave.within_range(extrema_final.get_x_image(), extrema_final.get_y_image(), closest_sigma_level, kernel_half_width)  {

            let dx = source_octave.dog_x_gradient[sigma_level].buffer[(extrema_final.get_y_image(),extrema_final.get_x_image())];
            let dy = source_octave.dog_y_gradient[sigma_level].buffer[(extrema_final.get_y_image(),extrema_final.get_x_image())];
            let ds = source_octave.dog_s_gradient[sigma_level].buffer[(extrema_final.get_y_image(),extrema_final.get_x_image())];

            let b = Matrix3x1::new(dx,dy,ds);
    

            let dog_sample = source_octave.difference_of_gaussians[closest_sigma_level].buffer.index((extrema_final.get_y_image(),extrema_final.get_x_image()));
            let dog_x_pertub = dog_sample + 0.5*b.dot(&perturb_final);
            (dog_x_pertub.abs(), extrema_final)
        } else {
            (-1.0, extrema_final)
        }

    

}

fn interpolate(source_octave: &SiftOctave, input_params: &dyn Feature, first_order_kernel: &dyn Kernel, second_order_kernel: &dyn Kernel) -> Matrix3x1<Float> {

    let sigma_level = input_params.get_closest_sigma_level();
    let dx = source_octave.dog_x_gradient[sigma_level].buffer[(input_params.get_y_image(),input_params.get_x_image())];
    let dy = source_octave.dog_y_gradient[sigma_level].buffer[(input_params.get_y_image(),input_params.get_x_image())];
    let ds = source_octave.dog_s_gradient[sigma_level].buffer[(input_params.get_y_image(),input_params.get_x_image())];

    let dxx = gradient_convolution_at_sample_3_d(&source_octave.difference_of_gaussians.iter().map(|x| x).collect(),input_params,second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample_3_d(&source_octave.difference_of_gaussians.iter().map(|x| x).collect(),input_params,second_order_kernel,GradientDirection::VERTICAL);
    let dss = gradient_convolution_at_sample_3_d(&source_octave.difference_of_gaussians.iter().map(|x| x).collect(),input_params,second_order_kernel,GradientDirection::SIGMA);

    let dxy = gradient_convolution_at_sample_3_d(&source_octave.dog_x_gradient.iter().map(|x| x).collect(),input_params,first_order_kernel,GradientDirection::VERTICAL);
    let dxs = gradient_convolution_at_sample_3_d(&source_octave.dog_x_gradient.iter().map(|x| x).collect(),input_params,first_order_kernel,GradientDirection::SIGMA);

    let dyx = gradient_convolution_at_sample_3_d(&source_octave.dog_y_gradient.iter().map(|x| x).collect(),input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let dys = gradient_convolution_at_sample_3_d(&source_octave.dog_y_gradient.iter().map(|x| x).collect(),input_params,first_order_kernel,GradientDirection::SIGMA);

    let dsx = gradient_convolution_at_sample_3_d(&source_octave.dog_s_gradient.iter().map(|x| x).collect(),input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let dsy = gradient_convolution_at_sample_3_d(&source_octave.dog_s_gradient.iter().map(|x| x).collect(),input_params,first_order_kernel,GradientDirection::VERTICAL);

    let a = Matrix3::new(dxx,dxy,dxs,
                         dyx,dyy,dys,
                         dsx,dsy,dss);

    let b = Matrix3x1::new(dx,dy,ds);
    let solve_option = (-a).qr().solve(&b); //.expect("Linear resolution failed.")
    match solve_option {
        Some(a) => a,
        None => panic!("Linear resolution failed.") 
    }
}


pub fn reject_edge_response_filter(source_octave: &SiftOctave, input_params: &SiftFeature, r: Float) -> bool {
    let hessian = harris_corner_matrix(source_octave,input_params);
    reject_edge(&hessian, r)
}

pub fn accept_edge_response_filter(source_octave: &SiftOctave, input_params: &SiftFeature, r: Float) -> bool {
    let hessian = harris_corner_matrix(source_octave,input_params);
    accept_edge(&hessian, r)
}