extern crate nalgebra as na;

use crate::Float;


pub mod orientation_histogram;

pub fn lagrange_interpolation_quadratic(a: Float, b: Float, c: Float, f_a: Float, f_b: Float, f_c: Float) -> Float {
    assert!( a < b && b < c);
    assert!( f_a > f_b && f_b < f_c);

    let numerator = (f_a-f_b)*(c-b).powi(2)-(f_c-f_b)*(b-a).powi(2);
    let denominator = (f_a-f_b)*(c-b)+(f_c-f_b)*(b-a);

    b + 0.5*(numerator/denominator)
}


