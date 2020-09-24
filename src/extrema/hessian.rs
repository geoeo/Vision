extern crate nalgebra as na;

use na::Matrix2;
use crate::pyramid::octave::Octave;
use crate::{Float, GradientDirection};
use crate::image::{kernel::Kernel,filter::gradient_convolution_at_sample};
use crate::extrema::extrema_parameters::ExtremaParameters;


//TODO: @Investigate: maybe precomputing the gradient images is more efficient
pub fn new(source_octave: &Octave, input_params: &ExtremaParameters, first_order_kernel: &dyn Kernel) -> Matrix2<Float> {


    let dxx = gradient_convolution_at_sample(source_octave,&source_octave.dog_x_gradient,input_params,first_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample(source_octave,&source_octave.dog_y_gradient,input_params,first_order_kernel,GradientDirection::VERTICAL);

    let dxy = gradient_convolution_at_sample(source_octave,&source_octave.dog_x_gradient,input_params,first_order_kernel,GradientDirection::VERTICAL);

    Matrix2::new(dxx,dxy,
        dxy,dyy)

}

pub fn accept_hessian(hessian: &Matrix2<Float>, r: Float) -> bool {
    let trace = hessian.trace();
    let determinant = hessian.determinant();
    let hessian_factor = trace.powi(2)/determinant;
    let r_factor = (r+1.0).powi(2)/r;

    hessian_factor < r_factor as Float && determinant > 0.0
}