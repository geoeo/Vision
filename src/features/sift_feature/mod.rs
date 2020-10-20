extern crate nalgebra as na;

use na::{Matrix3x1,Matrix3,DMatrix};

use crate::{Float,float, GradientDirection, round};
use crate::pyramid::{sift_octave::SiftOctave, runtime_params::RuntimeParams};
use std::fmt;


pub mod processing;

#[derive(Debug,Clone)]
pub struct SiftFeature {
    pub x: Float,
    pub y: Float,
    pub sigma_level: Float
    //pub sigma: Float
} 

impl fmt::Display for SiftFeature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}, s: {}", self.x, self.y, self.sigma_level)
    }
}

//TODO: check these
impl SiftFeature {
    pub fn x_image(&self) -> usize {
        self.x.trunc() as usize
    }

    pub fn y_image(&self) -> usize {
        self.y.trunc() as usize
    }

    pub fn closest_sigma_level(&self, s: usize) -> usize {
        // let sigma_range_half = (1.0/s as Float).exp2()/2.0;
        // let truncated_level =  self.sigma_level.trunc() as usize;
        // match self.sigma_level {
        //     level if level.fract() > sigma_range_half => truncated_level  + 1,
        //     _ => truncated_level
        // }
        self.sigma_level.trunc() as usize
    }
}


pub fn detect_sift_feature(source_octave: &SiftOctave, sigma_level: usize, x_step: usize, y_step: usize) -> Vec<SiftFeature> {

    let mut extrema_vec: Vec<SiftFeature> = Vec::new();

    assert!(sigma_level+1 < source_octave.difference_of_gaussians.len());
    assert!(sigma_level > 0);

    let image_buffer = &source_octave.difference_of_gaussians[sigma_level].buffer;
    let prev_buffer = &source_octave.difference_of_gaussians[sigma_level-1].buffer;
    let next_buffer = &source_octave.difference_of_gaussians[sigma_level+1].buffer;
    //let sigma = source_octave.sigmas[sigma_level];

    let offset = 5;

    for x in (offset..image_buffer.ncols()-offset).step_by(x_step) {
        for y in (offset..image_buffer.nrows()-offset).step_by(y_step)  {

            let sample_value = image_buffer[(y,x)];

            //TODO: @Investigate parallel
            let (is_smallest_curr, is_largest_curr) =  is_sample_extrema_in_neighbourhood(sample_value,x,y,image_buffer,true);
            let (is_smallest_prev, is_largest_prev) =  is_sample_extrema_in_neighbourhood(sample_value,x,y,prev_buffer,false);
            let (is_smallest_next, is_largest_next) = is_sample_extrema_in_neighbourhood(sample_value,x,y,next_buffer,false);

            //let is_extrema = (is_smallest_curr&&is_smallest_prev&&is_smallest_next) || (is_largest_curr&&is_largest_prev&&is_largest_next);
            let is_extrema = (is_smallest_curr||is_largest_curr) && (is_smallest_prev ||is_largest_prev) && (is_smallest_next || is_largest_next); // This is wrong ?

            if is_extrema {
                extrema_vec.push(SiftFeature{x: x as Float,y: y as Float,sigma_level: sigma_level as Float});
            }
        }
    }

    extrema_vec
}

fn is_sample_extrema_in_neighbourhood(sample: Float, x_sample: usize, y_sample: usize, neighbourhood_buffer: &DMatrix<Float>, skip_center: bool) -> (bool,bool) {

    let mut is_smallest = true;
    let mut is_largest = true;

    for x in x_sample-1..x_sample+2 {
        for y in y_sample-1..y_sample+2 {

            if x == x_sample && y == y_sample && skip_center {
                continue;
            }

            let value = neighbourhood_buffer[(y,x)];
            is_smallest &= sample < value;
            is_largest &= sample > value;

            if !(is_smallest || is_largest) {
                break;
            }

        }
    }

    (is_smallest,is_largest)
}

pub fn sift_feature_refinement(extrema: &Vec<SiftFeature>, source_octave: &SiftOctave,octave_level: usize, runtime_params: &RuntimeParams) -> Vec<SiftFeature> {
    extrema.iter().cloned().map(|x| processing::subpixel_refinement(source_octave,octave_level, &x)).filter(|x| x.0 >= runtime_params.contrast_r).map(|x| x.1).filter(|x| processing::reject_edge_response_filter(source_octave, &x, runtime_params.edge_r)).collect()
    //extrema.iter().cloned().filter(|x| accept_edge_response_filter(source_octave, &x, runtime_params.edge_r)).collect()
    //extrema.clone()
}





