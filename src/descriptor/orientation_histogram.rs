extern crate nalgebra as na;

use na::{Matrix1x2,Matrix2};

use crate::{float,Float};
use crate::image::Image;


#[derive(Debug,Clone)]
pub struct OrientationHistogram {
    max_val: Float,
    max_bin: usize,
    pub bins: Vec<Float>
}

impl OrientationHistogram {

    pub fn new(bin_size: usize) -> OrientationHistogram {
        OrientationHistogram{
            max_val: 0.0,
            max_bin: bin_size+1,
            bins: vec![0.0;bin_size]
        }
    }

}


pub fn gradient_and_orientation(x_gradient: &Image, y_gradient: &Image, x: usize, y: usize) -> (Float,Float) {

    let x_diff = x_gradient.buffer.index((y,x));
    let y_diff = y_gradient.buffer.index((y,x));

    let gradient = (x_diff.powi(2) + y_diff.powi(2)).sqrt();
    let orientation = (y_diff/x_diff).atan();

    (gradient,orientation)
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

//TODO check if this is missing a step -> Tthis is just for orientation!
pub fn add_measurement(histogram: &mut OrientationHistogram, grad_orientation: (Float,Float), weight: Float) -> () {
    let grad = grad_orientation.0;
    let orientation = grad_orientation.1;
    let bin_range = 2.0*float::consts::PI/(histogram.bins.len() as Float);
    let index = (orientation/bin_range).floor() as usize;

    histogram.bins[index] += grad*weight;

}

pub fn process_window(image: &Image,  x: usize, y: usize, sigma: Float) -> OrientationHistogram {

    let w = 3;
    for x_sample in x-w..x+w {
        for y_sample in y-w..y+w {

        }
    }

    OrientationHistogram::new(36)
}