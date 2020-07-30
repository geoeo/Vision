extern crate nalgebra as na;

use crate::{float,Float, ExtremaParameters, KeyPoint};
use crate::image::Image;
use crate::pyramid::octave::Octave;
use crate::descriptor::{lagrange_interpolation_quadratic,gauss_2d};


#[derive(Debug,Clone)]
pub struct OrientationHistogram {
    max_bin: usize,
    pub bin_range: Float,
    pub bins: Vec<Float>
}

impl OrientationHistogram {

    pub fn new(bin_len: usize) -> OrientationHistogram {
        OrientationHistogram{
            max_bin: bin_len,
            bin_range: 2.0*float::consts::PI/(bin_len as Float),
            bins: vec![0.0;bin_len]
        }
    }

    pub fn add_measurement(& mut self, grad_orientation: (Float,Float), weight: Float) -> () {
        let grad = grad_orientation.0;
        let orientation = grad_orientation.1;
        let index = (orientation/self.bin_range).floor() as usize;
        self.bins[index] += grad*weight;
    }

    pub fn index_to_radian(&self, index: Float) -> Float {
        assert!(index >=0.0 && index < self.bins.len() as Float);
        index*self.bin_range
    }

}


fn gradient_and_orientation(x_gradient: &Image, y_gradient: &Image, x: usize, y: usize) -> (Float,Float) {

    let x_diff = x_gradient.buffer.index((y,x));
    let y_diff = y_gradient.buffer.index((y,x));

    let gradient = (x_diff.powi(2) + y_diff.powi(2)).sqrt();
    let orientation = (y_diff/x_diff).atan();

    (gradient,orientation)
}


pub fn generate_keypoints_from_extrema(octave: &Octave, keypoint: &ExtremaParameters) -> Vec<KeyPoint> {

    let w = 3;
    let x = keypoint.x;
    let y = keypoint.y;
    let sigma = octave.sigmas[keypoint.sigma_level];
    let x_grad = &octave.x_gradient[keypoint.sigma_level]; //TODO: check 
    let y_grad = &octave.y_gradient[keypoint.sigma_level]; //TODO: check 
    let mut histogram = OrientationHistogram::new(36);

    //TODO: think of a better solution to image border
    //TODO: make image dimensions more easily accesible
    //TODO: reduce numbers of casts
    if x as isize -w < 0 || x +w as usize >= octave.images[0].buffer.ncols() || y as isize -w <0 || y+w as usize >= octave.images[0].buffer.nrows() {
        return Vec::new()
    }

    for x_sample in x-w as usize..x+w as usize {
        for y_sample in y-w as usize..y+w as usize {

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

    post_process(&histogram,keypoint)
}

fn post_process(histogram: &OrientationHistogram, extrema: &ExtremaParameters) -> Vec<KeyPoint> {

    let max_val = histogram.bins[histogram.max_bin];
    let threshold = max_val*0.8;
    let peaks_indices = histogram.bins.clone().into_iter().enumerate().filter(|x| x.1 >= threshold).map(|t| t.0).collect::<Vec<usize>>();
    let peak_neighbours_indices = peaks_indices.iter().map(|&x| get_cirular_closest(histogram, x)).collect::<Vec<(usize,usize,usize)>>();
    let interpolated_peaks_indices = peak_neighbours_indices.iter().map(|&(l,c,r)| 
        lagrange_interpolation_quadratic(l as Float,c as Float,r as Float,histogram.bins[l],histogram.bins[c],histogram.bins[r],0.0, (histogram.bins.len() - 1) as Float)
    ).collect::<Vec<Float>>();

    interpolated_peaks_indices.iter().map(|&peak_idx| {KeyPoint{x: extrema.x, y: extrema.y, sigma_level: extrema.sigma_level, orientation: histogram.index_to_radian(peak_idx)}}).collect::<Vec<KeyPoint>>()

}

fn get_cirular_closest(histogram: &OrientationHistogram, bin_idx: usize) -> (usize,usize,usize) {
    let bin_len = histogram.bins.len();
    assert!(bin_len >=3);

    match bin_idx {
        idx if idx == 0 => (bin_len-1,idx,1),
        idx if idx == bin_len - 1 => (idx-1,idx,0),
        idx => (idx - 1,idx, idx + 1)
    }
    
}

