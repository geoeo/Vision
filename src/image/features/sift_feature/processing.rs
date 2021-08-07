extern crate nalgebra as na;

use na::{Matrix3x1, Matrix3};
use crate::image::pyramid::sift::sift_octave::SiftOctave;
use crate::{Float, GradientDirection};
use crate::image::filter::{kernel::Kernel,gradient_convolution_at_sample,prewitt_kernel::PrewittKernel,laplace_kernel::LaplaceKernel};
use crate::image::features::{Feature,sift_feature::SiftFeature};


//TODO: maybe return new extrema instead due to potential change of coordiantes in interpolation
//TODO: needs to be more stable
pub fn subpixel_refinement(source_octave: &SiftOctave, feature:  &SiftFeature) -> (Float,SiftFeature) {

    let sigma_level = feature.get_closest_sigma_level();

    let first_order_kernel = PrewittKernel::new();
    let second_order_kernel = LaplaceKernel::new();


    let kernel_half_width = first_order_kernel.radius();

    let s = source_octave.s();
    let sigma_range = (1.0/(s as Float)).exp2();
    let mut perturb_final = interpolate(source_octave,feature,&first_order_kernel,&second_order_kernel);
    let mut extrema_final =  SiftFeature{x:feature.x ,y:feature.y,sigma_level:feature.sigma_level, id: None};
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

fn interpolate(source_octave: &SiftOctave, feature: &dyn Feature, first_order_kernel: &dyn Kernel, second_order_kernel: &dyn Kernel) -> Matrix3x1<Float> {

    let sigma_level = feature.get_closest_sigma_level();
    let dx = source_octave.dog_x_gradient[sigma_level].buffer[(feature.get_y_image(),feature.get_x_image())];
    let dy = source_octave.dog_y_gradient[sigma_level].buffer[(feature.get_y_image(),feature.get_x_image())];
    let ds = source_octave.dog_s_gradient[sigma_level].buffer[(feature.get_y_image(),feature.get_x_image())];

    let dxx = gradient_convolution_at_sample(&source_octave.difference_of_gaussians,feature,second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample(&source_octave.difference_of_gaussians,feature,second_order_kernel,GradientDirection::VERTICAL);
    let dss = gradient_convolution_at_sample(&source_octave.difference_of_gaussians,feature,second_order_kernel,GradientDirection::SIGMA);

    let dxy = gradient_convolution_at_sample(&source_octave.dog_x_gradient,feature,first_order_kernel,GradientDirection::VERTICAL);
    let dxs = gradient_convolution_at_sample(&source_octave.dog_x_gradient,feature,first_order_kernel,GradientDirection::SIGMA);

    let dyx = gradient_convolution_at_sample(&source_octave.dog_y_gradient,feature,first_order_kernel,GradientDirection::HORIZINTAL);
    let dys = gradient_convolution_at_sample(&source_octave.dog_y_gradient,feature,first_order_kernel,GradientDirection::SIGMA);

    let dsx = gradient_convolution_at_sample(&source_octave.dog_s_gradient,feature,first_order_kernel,GradientDirection::HORIZINTAL);
    let dsy = gradient_convolution_at_sample(&source_octave.dog_s_gradient,feature,first_order_kernel,GradientDirection::VERTICAL);

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

