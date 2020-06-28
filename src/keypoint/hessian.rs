extern crate nalgebra as na;

use na::Matrix2;
use crate::pyramid::octave::Octave;
use crate::{Float,ExtremaParameters, GradientDirection};
use crate::image::{kernel::Kernel,filter::gradient_eval};


//TODO: @Investigate: maybe precomputing the gradient images is more efficient
pub fn new(source_octave: &Octave, input_params: &ExtremaParameters,  first_order_kernel: &dyn Kernel, second_order_kernel: &dyn Kernel) -> Matrix2<Float> {

    let extrema_top = ExtremaParameters{x: input_params.x, y: input_params.y - 1, sigma_level: input_params.sigma_level};
    let extrema_bottom = ExtremaParameters{x: input_params.x, y: input_params.y + 1, sigma_level: input_params.sigma_level};
    let extrema_left = ExtremaParameters{x: input_params.x - 1, y: input_params.y, sigma_level: input_params.sigma_level};
    let extrema_right = ExtremaParameters{x: input_params.x + 1, y: input_params.y, sigma_level: input_params.sigma_level};


    let dx = gradient_eval(source_octave,&input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let dx_top = gradient_eval(source_octave,&extrema_top,first_order_kernel,GradientDirection::HORIZINTAL);
    let dx_bottom = gradient_eval(source_octave,&extrema_bottom,first_order_kernel,GradientDirection::HORIZINTAL);
    let dy =  gradient_eval(source_octave,&input_params,first_order_kernel,GradientDirection::VERTICAL);
    let dy_left = gradient_eval(source_octave,&extrema_left,first_order_kernel,GradientDirection::VERTICAL);
    let dy_right = gradient_eval(source_octave,&extrema_right,first_order_kernel,GradientDirection::VERTICAL);

    let dxx = gradient_eval(source_octave,input_params,second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_eval(source_octave,input_params,second_order_kernel,GradientDirection::VERTICAL);

    let dxy = dx_top - 2.0*dx + dx_bottom; 
    let dyx = dy_left - 2.0*dy + dy_right; 

    Matrix2::new(dxx,dxy,
                 dyx,dyy)

}

pub fn eval_hessian(hessian: &Matrix2<Float>, r: usize) -> bool {
    let trace = hessian.trace();
    let determinant = hessian.determinant();
    let hessian_factor = trace.powi(2)/determinant;
    let r_factor = (r+1).pow(2)/r;

    hessian_factor < r_factor as Float
}