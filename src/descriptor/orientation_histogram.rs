extern crate nalgebra as na;

use crate::{float,Float, ExtremaParameters, KeyPoint};
use crate::pyramid::octave::Octave;
use crate::descriptor::{lagrange_interpolation_quadratic, gauss_2d, gradient_and_orientation};


#[derive(Debug,Clone)]
pub struct OrientationHistogram {
    max_bin: usize,
    pub squared_magnitude: Float,
    pub bin_range: Float,
    pub bins: Vec<Float>
}

impl OrientationHistogram {

    pub fn new(bin_len: usize) -> OrientationHistogram {
        OrientationHistogram{
            max_bin: bin_len,
            squared_magnitude: 0.0,
            bin_range: 2.0*float::consts::PI/(bin_len as Float),
            bins: vec![0.0;bin_len]
        }
    }

    pub fn add_measurement(&mut self, grad_orientation: (Float,Float), weight: Float) -> () {
        let grad = grad_orientation.0;
        let orientation = grad_orientation.1;
        let index = radian_to_index(self,orientation);
        let v = grad*weight;
        self.bins[index] += v;
        self.squared_magnitude = v.powi(2);
    }


    pub fn add_measurement_to_adjecent_with_interp(&mut self, grad_orientation: (Float,Float), main_orientation: Float, weight: Float) -> () {
        let grad = grad_orientation.0;
        let orientation = grad_orientation.1;
        let index = radian_to_index(self,orientation);
        let main_index = radian_to_index(self,main_orientation) as isize;

        let(l,c,r) = get_cirular_adjecent(self,index as isize);
        let l_weight = (main_index - l as isize).abs();
        let c_weight = (main_index - c as isize).abs();
        let r_weight = (main_index - r as isize).abs();

        let v_l = grad*weight*l_weight as Float;
        let v_c = grad*weight*c_weight as Float;
        let v_r = grad*weight*r_weight as Float;

        self.bins[l] += v_l;
        self.bins[c] += v_c;
        self.bins[r] += v_r;

        self.squared_magnitude = v_l.powi(2) + v_c.powi(2) + v_r.powi(2);
    }

    pub fn add_histogram(&mut self, other_histogram: &OrientationHistogram, weight: Float) -> () {

        for i in 0..self.bins.len() {
            let value = other_histogram.bins[i];
            self.bins[i] += value*weight;
        }

    }

}

pub fn index_to_radian(histogram: &OrientationHistogram, index: Float) -> Float {
    assert!(index >=0.0 && index < histogram.bins.len() as Float);
    index*histogram.bin_range
}

pub fn radian_to_index(histogram: &OrientationHistogram, orientation: Float) -> usize {
    (orientation/histogram.bin_range).floor() as usize
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
            let grad_orientation = gradient_and_orientation(x_grad, y_grad, x_sample, y_sample);
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

    let peak_neighbours_indices = peaks_indices.iter().map(|&x| get_cirular_adjecent(histogram, x as isize)).collect::<Vec<(usize,usize,usize)>>();
    let interpolated_peaks_indices = peak_neighbours_indices.iter().map(|&(l,c,r)| 
        lagrange_interpolation_quadratic(l as Float,c as Float,r as Float,histogram.bins[l],histogram.bins[c],histogram.bins[r],0.0, (histogram.bins.len()) as Float)
    ).collect::<Vec<Float>>();


    interpolated_peaks_indices.iter().map(|&peak_idx| {KeyPoint{x: extrema.x, y: extrema.y, sigma_level: extrema.sigma_level, orientation: index_to_radian(histogram,peak_idx)}}).collect::<Vec<KeyPoint>>()

}

pub fn get_cirular_adjecent(histogram: &OrientationHistogram, bin_idx: isize) -> (usize,usize,usize) {
    let bin_len = histogram.bins.len() as isize;
    assert!(bin_len >=3);

    let mut left_found = false;
    let mut left_final_idx = 0;
    let mut right_found = false;
    let mut right_final_idx = 0;

    let center_value = histogram.bins[bin_idx as usize];

    for i in 0..bin_len {

        if !left_found {
            let left_index = match bin_idx -i {
                idx if idx < 0 => bin_len + idx,
                idx => idx
            };
    
            let left_value = histogram.bins[left_index as usize];
    
            if left_value < center_value {
                left_final_idx = left_index;
                left_found = true;
            }
        }

        if !right_found {
            let right_index = match bin_idx + i {
                idx if idx >= bin_len => idx - bin_len,
                idx => idx 
            };
            let right_value = histogram.bins[right_index as usize];
    
            if right_value < center_value {
                right_final_idx = right_index;
                right_found = true;
            }
        }

        if left_found && right_found {
            break;
        }

    };

    (left_final_idx as usize,bin_idx as usize,right_final_idx as usize)

}


