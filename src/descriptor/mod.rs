extern crate nalgebra as na;

use crate::Float;


pub mod orientation_histogram;

pub fn lagrange_interpolation_quadratic(a: Float, b: Float, c: Float, f_a: Float, f_b: Float, f_c: Float, range_min: Float, range_max: Float) -> Float {

    let a_corrected = if a > b { a - range_max} else {a};
    let c_corrected = if b > c { c + range_max} else {c};

    assert!( a < b && b < c);
    assert!( f_a > f_b && f_b < f_c);

    let numerator = (f_a-f_b)*(c_corrected-b).powi(2)-(f_c-f_b)*(b-a_corrected).powi(2);
    let denominator = (f_a-f_b)*(c_corrected-b)+(f_c-f_b)*(b-a_corrected);

    let result  = b + 0.5*(numerator/denominator);

    match result {
        res if res < range_min => res + range_max,
        res if res > range_max => res - range_max,
        res => res
    }
}


