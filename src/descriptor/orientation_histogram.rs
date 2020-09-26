extern crate nalgebra as na;

use crate::{float,Float};
use crate::pyramid::{octave::Octave,runtime_params::RuntimeParams};
use crate::descriptor::{lagrange_interpolation_quadratic,quadatric_interpolation, gauss_2d, gradient_and_orientation, keypoint::KeyPoint};
//use crate::ORIENTATION_HISTOGRAM_WINDOW_SIZE;
use crate::extrema::extrema_parameters::ExtremaParameters;


#[derive(Debug,Clone)]
pub struct OrientationHistogram {
    max_bin: usize, // not set for descriptor histograms
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

        //let(l,c,r) = get_adjacent_circular_by_value(self,index as isize);
        let(l,c,r) = get_adjacent_circular_by_index(self,index as isize);
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

    pub fn get_value_bounded(&self, i: isize) -> Float {
        let len = self.bins.len() as isize;
        match i {
            idx if idx <= 0 => self.bins[0],
            idx if idx >= len => self.bins[(len-1) as usize],
            idx => self.bins[idx as usize]
        }
    }

    pub fn smooth(self: &mut OrientationHistogram) -> () {
        for i in 0..self.bins.len() {
            let idx = i as isize;
            self.bins[i] = 
                (self.get_value_bounded(idx-2) + self.get_value_bounded(idx+2))/16.0 + 
                (self.get_value_bounded(idx-1) + self.get_value_bounded(idx+1))*4.0/16.0 + 
                self.get_value_bounded(idx)*6.0/16.0
        }
    }

}

pub fn index_to_radian(histogram: &OrientationHistogram, index: Float) -> Float {
    assert!(index >=0.0 && index < histogram.bins.len() as Float);
    index*histogram.bin_range
}

pub fn radian_to_index(histogram: &OrientationHistogram, orientation: Float) -> usize {
    (orientation/histogram.bin_range).trunc() as usize
}

pub fn generate_keypoints_from_extrema(octave: &Octave,octave_level: usize, keypoint: &ExtremaParameters, runtime_params: &RuntimeParams) -> Vec<KeyPoint> {


    let x = keypoint.x;
    let y = keypoint.y;
    let sigma = octave.sigmas[keypoint.sigma_level];
    let new_sigma = 1.5*octave.sigmas[keypoint.sigma_level];
    let w = (runtime_params.orientation_histogram_window_factor as Float * new_sigma).trunc() as isize;
    //let w = (runtime_params.orientation_histogram_window_factor as Float * sigma).trunc() as isize;
    let x_grad = &octave.x_gradient[keypoint.sigma_level]; //TODO: check 
    let y_grad = &octave.y_gradient[keypoint.sigma_level]; //TODO: check 
    let mut histogram = OrientationHistogram::new(36);
    let inter_pixel_distance = Octave::inter_pixel_distance(octave_level);

    //TODO: think of a better solution to image border
    //TODO: make image dimensions more easily accesible
    //TODO: reduce numbers of casts
    if x as isize -w < 0 || 
    x +w as usize >= octave.images[keypoint.sigma_level].buffer.ncols() || 
    y as isize -w <0 || 
    y+w as usize >= octave.images[keypoint.sigma_level].buffer.nrows(){
        return Vec::new()
    }

    //TODO: extra filter for max 

    for x_sample in x-w as usize..x+w as usize {
        for y_sample in y-w as usize..y+w as usize {
            let x_corrected = (inter_pixel_distance * x_sample as Float);
            let y_corrected = (inter_pixel_distance * y_sample as Float);
            let gauss_weight = gauss_2d(x as Float, y as Float, x_corrected,y_corrected, new_sigma); //TODO: maybe precompute this
            //let gauss_weight = gauss_2d(x as Float, y as Float,  x_sample as Float, y_sample as Float, new_sigma); //TODO: maybe precompute this
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

    //TODO: maybe smooth here
    post_process(&mut histogram,keypoint, octave_level)
}


fn post_process(histogram: &mut OrientationHistogram, extrema: &ExtremaParameters,octave_level: usize) -> Vec<KeyPoint> {

    let max_val = histogram.bins[histogram.max_bin];
    let threshold = max_val*0.8;
    let peaks_indices = histogram.bins.clone().into_iter().enumerate().filter(|x| x.1 >= threshold).map(|t| t.0).collect::<Vec<usize>>();

    let peak_neighbours_indices = peaks_indices.into_iter().filter(|&x| filter_adjacent(histogram,x as isize)).map(|x| get_adjacent_circular_by_index(histogram, x as isize)).collect::<Vec<(usize,usize,usize)>>();
    let interpolated_peaks_indices = peak_neighbours_indices.iter().map(|&(l,c,r)| 
        lagrange_interpolation_quadratic(l as Float,c as Float,r as Float,histogram.bins[l],histogram.bins[c],histogram.bins[r],0.0, (histogram.bins.len()) as Float)
    ).collect::<Vec<Float>>();

    //histogram.smooth();
    //TODO: maybe split up the return of histogram and keypoint so that it can be debugged
    interpolated_peaks_indices.iter().map(|&peak_idx| {KeyPoint{x: extrema.x, y: extrema.y, octave_level: octave_level, sigma_level: extrema.sigma_level, orientation: index_to_radian(histogram,peak_idx)}}).collect::<Vec<KeyPoint>>()

}

pub fn filter_adjacent(histogram: &OrientationHistogram, bin_idx: isize) -> bool {
    let (l,c,r) = get_adjacent_circular_by_index(histogram, bin_idx);
    let c_val = histogram.bins[c];
    return c_val > histogram.bins[l] && c_val > histogram.bins[r]
}

fn get_adjacent_circular_by_index(histogram: &OrientationHistogram, bin_idx: isize) -> (usize,usize,usize) {
    let bin_len = histogram.bins.len() as isize;
    assert!(bin_len >=3);

    let l = match bin_idx - 1 {
        idx if idx < 0 => bin_len-1,
        idx => idx
    };

    let r = match bin_idx + 1 {
        idx if idx == bin_len => 0,
        idx => idx
    };

    (l as usize, bin_idx as usize, r as usize)
}

// pub fn get_adjacent_circular_by_value(histogram: &OrientationHistogram, bin_idx: isize) -> (usize,usize,usize) {
//     let bin_len = histogram.bins.len() as isize;
//     assert!(bin_len >=3);

//     let mut left_found = false;
//     let mut left_final_idx = 0;
//     let mut right_found = false;
//     let mut right_final_idx = 0;

//     let center_value = histogram.bins[bin_idx as usize];

//     for i in 0..bin_len {

//         if !left_found {
//             let left_index = match bin_idx -i {
//                 idx if idx < 0 => bin_len + idx,
//                 idx => idx
//             };
    
//             let left_value = histogram.bins[left_index as usize];
    
//             if left_value < center_value {
//                 left_final_idx = left_index;
//                 left_found = true;
//             }
//         }

//         if !right_found {
//             let right_index = match bin_idx + i {
//                 idx if idx >= bin_len => idx - bin_len,
//                 idx => idx 
//             };
//             let right_value = histogram.bins[right_index as usize];
    
//             if right_value < center_value {
//                 right_final_idx = right_index;
//                 right_found = true;
//             }
//         }

//         if left_found && right_found {
//             break;
//         }

//     };

//     (left_final_idx as usize,bin_idx as usize,right_final_idx as usize)

// }




