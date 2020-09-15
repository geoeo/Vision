extern crate nalgebra as na;

use na::{Matrix1x2,Matrix2,Matrix3,Matrix3x1,linalg::LU};
use crate::{float,Float,round};
use crate::image::Image;

pub mod orientation_histogram;
pub mod local_image_descriptor;
pub mod feature_vector;

pub const DESCRIPTOR_BINS: usize = 16;
pub const ORIENTATION_BINS: usize = 8;

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
    assert!(f_a <= f_b && f_b >= f_c ); 

    let numerator = (f_a-f_b)*(c_corrected-b).powi(2)-(f_c-f_b)*(b-a_corrected).powi(2);
    let denominator = (f_a-f_b)*(c_corrected-b)+(f_c-f_b)*(b-a_corrected);

    let result  = b + 0.5*(numerator/denominator);

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

    let decomp = a.lu();
    let x = decomp.solve(&b).expect("Linear resolution failed.");

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
    let sigma_recip = 1.0/sigma;
    let covariance = Matrix2::new(sigma_recip, 0.0,0.0, sigma_recip);


    let exponent = -0.5*offset*(covariance*offset_transpose);
    let exp = exponent.index((0,0)).exp();

    let det = sigma.powi(2);
    let denom = 2.0*float::consts::PI*det.sqrt();

    exp/denom
}

pub fn rotation_matrix_2d_from_orientation(orientation: Float) -> Matrix2<Float> {

    Matrix2::new(orientation.cos(), -orientation.sin(),
                orientation.sin(), orientation.cos())

}

pub fn gradient_and_orientation(x_gradient: &Image, y_gradient: &Image, x: usize, y: usize) -> (Float,Float) {

    let x_diff = round(x_gradient.buffer.index((y,x)).clone(),5);
    let y_diff = round(y_gradient.buffer.index((y,x)).clone(),5);

    let gradient = (x_diff.powi(2) + y_diff.powi(2)).sqrt();
    let orientation = match  y_diff.atan2(x_diff.clone()) {
        angle if angle < 0.0 => 2.0*float::consts::PI + angle,
        angle => angle
    };

    (gradient,orientation)
}


