extern crate nalgebra as na;

use na::{Matrix2,Matrix1x2,Matrix3x1, Matrix3};
use crate::pyramid::sift_octave::SiftOctave;
use crate::{float,Float, GradientDirection};
use crate::filter::{kernel::Kernel,gradient_convolution_at_sample,prewitt_kernel::PrewittKernel,laplace_kernel::LaplaceKernel,laplace_off_center_kernel::LaplaceOffCenterKernel};
use crate::feature::sift_feature::SiftFeature;


//TODO: @Investigate: maybe precomputing the gradient images is more efficient
pub fn new(source_octave: &SiftOctave, input_params: &SiftFeature) -> Matrix2<Float> {

    let second_order_kernel = LaplaceKernel::new();
    let first_order_kernel = PrewittKernel::new();


    let dxx = gradient_convolution_at_sample(source_octave,&source_octave.difference_of_gaussians,input_params,&second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample(source_octave,&source_octave.difference_of_gaussians,input_params,&second_order_kernel,GradientDirection::VERTICAL);
    let dxy = gradient_convolution_at_sample(source_octave,&source_octave.dog_x_gradient,input_params,&first_order_kernel,GradientDirection::VERTICAL);

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
pub fn subpixel_refinement(source_octave: &SiftOctave, octave_level: usize, input_params: &SiftFeature) -> (Float,SiftFeature) {

    let sigma_level = input_params.closest_sigma_level(source_octave.s());

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

        let closest_sigma_level = extrema_final.closest_sigma_level(source_octave.s());

        if source_octave.within_range(extrema_final.x_image(), extrema_final.y_image(), closest_sigma_level, kernel_half_width) {
            perturb_final = interpolate(source_octave,&extrema_final, &first_order_kernel,&second_order_kernel);
            counter = counter +1;

        } else {
            counter = max_it;
        }

    }


    let closest_sigma_level = extrema_final.closest_sigma_level(source_octave.s());
    if source_octave.within_range(extrema_final.x_image(), extrema_final.y_image(), closest_sigma_level, kernel_half_width)  {

            let dx = source_octave.dog_x_gradient[sigma_level].buffer[(extrema_final.y_image(),extrema_final.x_image())];
            let dy = source_octave.dog_y_gradient[sigma_level].buffer[(extrema_final.y_image(),extrema_final.x_image())];
            let ds = source_octave.dog_s_gradient[sigma_level].buffer[(extrema_final.y_image(),extrema_final.x_image())];

            let b = Matrix3x1::new(dx,dy,ds);
    

            let dog_sample = source_octave.difference_of_gaussians[closest_sigma_level].buffer.index((extrema_final.y_image(),extrema_final.x_image()));
            let dog_x_pertub = dog_sample + 0.5*b.dot(&perturb_final);
            (dog_x_pertub.abs(), extrema_final)
        } else {
            (-1.0, extrema_final)
        }

    

}

fn interpolate(source_octave: &SiftOctave, input_params: &SiftFeature, first_order_kernel: &dyn Kernel, second_order_kernel: &dyn Kernel) -> Matrix3x1<Float> {

    let sigma_level = input_params.closest_sigma_level(source_octave.s());
    let dx = source_octave.dog_x_gradient[sigma_level].buffer[(input_params.y_image(),input_params.x_image())];
    let dy = source_octave.dog_y_gradient[sigma_level].buffer[(input_params.y_image(),input_params.x_image())];
    let ds = source_octave.dog_s_gradient[sigma_level].buffer[(input_params.y_image(),input_params.x_image())];

    let dxx = gradient_convolution_at_sample(source_octave,&source_octave.difference_of_gaussians,input_params,second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample(source_octave,&source_octave.difference_of_gaussians,input_params,second_order_kernel,GradientDirection::VERTICAL);
    let dss = gradient_convolution_at_sample(source_octave,&source_octave.difference_of_gaussians,input_params,second_order_kernel,GradientDirection::SIGMA);

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
    let solve_option = (-a).qr().solve(&b); //.expect("Linear resolution failed.")
    match solve_option {
        Some(a) => a,
        None => panic!("Linear resolution failed.") 
    }
}

//TODO: Doesnt seem to work as well as lagrange -> produces out  of scope results
pub fn newton_interpolation_quadratic(a: Float, b: Float, c: Float, f_a: Float, f_b: Float, f_c: Float, range_min: Float, range_max: Float) -> Float {

    let a_corrected = if a > b { a - range_max} else {a};
    let c_corrected = if b > c { c + range_max} else {c};

    assert!( a_corrected < b && b < c_corrected);
    assert!(f_a <= f_b && f_b >= f_c ); 

    let b_2 = (f_b - f_c)/(b-c_corrected);
    let b_3 = (((f_c - f_b)/(c_corrected-b))-((f_b-f_a)/(b-a_corrected)))/(c_corrected-a_corrected);

    let result  = (-b_2 + a_corrected + b) / (2.0*b_3);

    match result {
        res if res < range_min => res + range_max,
        res if res > range_max => res - range_max,
        res => res
    }

}


// http://fourier.eng.hmc.edu/e176/lectures/NM/node25.html
pub fn lagrange_interpolation_quadratic(a: Float, b: Float, c: Float, f_a: Float, f_b: Float, f_c: Float, range_min: Float, range_max: Float) -> Float {

    let a_corrected = if a > b { a - range_max} else {a};
    let c_corrected = if b > c { c + range_max} else {c};

    assert!( a_corrected < b && b < c_corrected);
    assert!(f_b >= f_a  && f_b >= f_c ); 

    let numerator = (f_a-f_b)*(c_corrected-b).powi(2)-(f_c-f_b)*(b-a_corrected).powi(2);
    let denominator = (f_a-f_b)*(c_corrected-b)+(f_c-f_b)*(b-a_corrected);

    let mut result  = b + 0.5*(numerator/denominator);
    //result += 5.0; // extra 5 to move the orientation into the center of the bin

    match result {
        res if res < range_min => res + range_max,
        res if res >= range_max => res - range_max,
        res => res
    }
}

// https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
pub fn quadatric_interpolation(a: Float, b: Float, c: Float, f_a: Float, f_b: Float, f_c: Float, range_min: Float, range_max: Float) -> Float {

    let a = Matrix3::new(a.powi(2),a,1.0,
                         b.powi(2),b,1.0,
                         c.powi(2),c,1.0);
    let b = Matrix3x1::new(f_a,f_b,f_c);

    let x = a.lu().solve(&b).expect("Linear resolution failed.");

    let coeff_a = x[(0,0)];
    let coeff_b = x[(1,0)];


    let result = -coeff_b/(2.0*coeff_a);

    match result {
        res if res < range_min => res + range_max,
        res if res >= range_max => res - range_max,
        res => res
    }
}

pub fn gauss_2d(x_center: Float, y_center: Float, x: Float, y: Float, sigma: Float) -> Float {
    let offset = Matrix1x2::new(x-x_center,y-y_center);
    let offset_transpose = offset.transpose();
    let sigma_sqr = sigma.powi(2);
    let sigma_sqr_recip = 1.0/sigma_sqr;
    let covariance = Matrix2::new(sigma_sqr_recip, 0.0,0.0, sigma_sqr_recip);


    let exponent = -0.5*offset*(covariance*offset_transpose);
    let exp = exponent.index((0,0)).exp();


    let denom = 2.0*float::consts::PI*sigma_sqr;

    exp/denom
}