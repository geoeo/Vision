extern crate nalgebra as na;

use na::{Matrix1x2,Matrix2};

use crate::{float,Float, ExtremaParameters, KeyPoint};
use crate::image::Image;
use crate::pyramid::octave::Octave;


#[derive(Debug,Clone)]
pub struct OrientationHistogram {
    max_bin: usize,
    pub bins: Vec<Float>
}

impl OrientationHistogram {

    pub fn new(bin_size: usize) -> OrientationHistogram {
        OrientationHistogram{
            max_bin: bin_size+1,
            bins: vec![0.0;bin_size]
        }
    }

    //TODO check if this is missing a step -> Tthis is just for orientation!
    pub fn add_measurement(& mut self, grad_orientation: (Float,Float), weight: Float) -> () {
        let grad = grad_orientation.0;
        let orientation = grad_orientation.1;
        let bin_range = 2.0*float::consts::PI/(self.bins.len() as Float);
        let index = (orientation/bin_range).floor() as usize;

        self.bins[index] += grad*weight;

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


pub fn process_window(octave: &Octave, keypoint: &ExtremaParameters) -> Vec<KeyPoint> {

    let w = 3;
    let x = keypoint.x;
    let y = keypoint.y;
    let sigma = octave.sigmas[keypoint.sigma_level];
    let x_grad = &octave.x_gradient[keypoint.sigma_level]; //TODO: check 
    let y_grad = &octave.y_gradient[keypoint.sigma_level]; //TODO: check 
    let mut histogram = OrientationHistogram::new(36);

    for x_sample in x-w..x+w {
        for y_sample in y-w..y+w {

            let new_sigma = 1.5*sigma;
            let gauss_weight = gauss_2d(x as Float, y as Float, x_sample as Float, y_sample as Float, new_sigma); //TODO: maybe precompute this
            let grad_orientation = gradient_and_orientation(x_grad, y_grad, x, y);
            histogram.add_measurement(grad_orientation, gauss_weight);
        }
    }

    let mut iter = histogram.bins.iter().enumerate();

    let init = iter.next().unwrap();
    
    histogram.max_bin = iter.fold(init, |acc, x| {
        let cmp = x.1.partial_cmp(acc.1).unwrap();
        let max = if let std::cmp::Ordering::Greater = cmp {
            x
        } else {
            acc
        };
        max
    }).0;

    post_process(&histogram)
}

fn post_process(histogram: &OrientationHistogram) ->  Vec<KeyPoint> {
    //TODO: peak post processing
    Vec::with_capacity(2)

}