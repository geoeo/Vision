extern crate nalgebra as na;

use na::Matrix2;
use crate::pyramid::octave::Octave;
use crate::{Float,ExtremaParameters, GradientDirection};
use crate::image::{kernel::Kernel,filter::gradient_eval};


//TODO: @Investigate: maybe precomputing the gradient images is more efficient
pub fn new(source_octave: &Octave, input_params: &ExtremaParameters,  first_order_kernel: &dyn Kernel, second_order_kernel: &dyn Kernel) -> Matrix2<Float> {

    let center = (input_params.y,input_params.x);
    let left = (input_params.y,input_params.x-1);
    let right = (input_params.y,input_params.x+1);
    let top = (input_params.y-1,input_params.x);
    let bottom = (input_params.y+1,input_params.x);


    let x_gradient_image = &source_octave.x_gradient[input_params.sigma_level];
    let y_gradient_image = &source_octave.y_gradient[input_params.sigma_level];

    let dx = x_gradient_image.buffer.index(center);
    let dx_top = x_gradient_image.buffer.index(top);
    let dx_bottom = x_gradient_image.buffer.index(bottom);
    let dy =  y_gradient_image.buffer.index(center);
    let dy_left = y_gradient_image.buffer.index(left);
    let dy_right = y_gradient_image.buffer.index(right);

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