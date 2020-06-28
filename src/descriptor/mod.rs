extern crate nalgebra as na;

use na::Matrix2;
use crate::pyramid::octave::Octave;
use crate::{Float,ExtremaParameters, GradientDirection};
use crate::image::{Image, kernel::Kernel,filter::gradient_eval};

pub fn lagrange_interpolation_quadratic(a: Float, b: Float, c: Float, f_a: Float, f_b: Float, f_c: Float) -> Float {
    assert!( a < b && b < c);
    assert!( f_a > f_b && f_b < f_c);

    let numerator = (f_a-f_b)*(c-b).powi(2)-(f_c-f_b)*(b-a).powi(2);
    let denominator = (f_a-f_b)*(c-b)+(f_c-f_b)*(b-a);

    b + 0.5*(numerator/denominator)
}

pub fn gradient_and_orientation(x_gradient: &Image, y_gradient: &Image, x: usize, y: usize) -> (Float,Float) {

    let x_diff = x_gradient.index((y,x));
    let y_diff = y_gradient.index((y,x));

    let gradient = (x_diff.powi(2) + y_diff.powi(2)).sqrt();
    let orientation = (y_diff/x_diff).atan();

    (gradient,orientation)
}
