extern crate nalgebra as na;

use na::{Matrix2,Matrix3,Matrix1x2,Matrix3x1, Vector,Dim, storage::Storage,DVector };
use crate::image::Image;
use crate::{Float,float};

pub mod lie;
pub mod pose;
pub mod loss;
pub mod solver;
pub mod weighting;

pub fn quadratic_roots(a: Float, b: Float, c: Float) -> (Float,Float) {
    let det = b.powi(2)-4.0*a*c;
    match det {
        det if det > 0.0 => {
            let det_sqrt = det.sqrt();
            ((-b - det_sqrt)/2.0*a,(-b + det_sqrt)/2.0*a)
        },
        det if det < 0.0 => panic!("determinat less than zero, no real solutions"),
        _ => {
            let res = -b/2.0*a;
            (res,res)
        }

    }
}

pub fn median_absolute_deviation(data: &DVector<Float>) -> Float {
    let (median_value, sorted_data) = median(data.data.as_vec().clone(), true);
    let absolute_deviation: Vec<Float> = sorted_data.iter().map(|x| (x-median_value).abs()).collect();
    median(absolute_deviation,false).0
}

pub fn median(data: Vec<Float>, sort_data: bool) -> (Float, Vec<Float>) {
    let mut mut_data = data;
    if sort_data {
        mut_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let middle = mut_data.len()/2;
    (mut_data[middle], mut_data)
}

pub fn round(number: Float, dp: i32) -> Float {
    let n = (10.0 as Float).powi(dp);
    (number * n).round()/n
}

pub fn calc_sigma_from_z(z: Float, x: Float, mean: Float) -> Float {
    assert!(z > 0.0);
    (x-mean)/z
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

    let result  = b + 0.5*(numerator/denominator);
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

pub fn max_norm<D,S>(vector: &Vector<Float,D,S>) -> Float where D: Dim, S: Storage<Float,D> {

    vector.iter().fold(0.0,|max,v| 
        match v.abs() {
            v_abs if v_abs > max => v_abs,
            _ => max
        }
    )

}










