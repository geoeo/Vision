extern crate nalgebra as na;

use na::DMatrix;

use crate::Float;
use crate::image::pyramid::sift::{sift_runtime_params::SiftRuntimeParams,sift_octave::SiftOctave};
use crate::image::features::{Feature,hessian_response, geometry::point::Point,};
use std::{fmt, hash::{Hasher,Hash}};


pub mod processing;

#[derive(Debug,Clone)]
pub struct SiftFeature {
    pub x: Float,
    pub y: Float,
    pub sigma_level: Float,
    pub id: Option<u64>
} 

impl Feature for SiftFeature {
    fn new(x: Float, y: Float, landmark_id: Option<usize>) -> SiftFeature { panic!("TODO: SiftFeature new") }
    fn get_location(&self) -> Point<Float> { Point::<Float> { x: self.get_x_image_float(), y: self.get_y_image_float() } }
    fn get_x_image_float(&self) -> Float { self.get_x_image() as Float}
    fn get_y_image_float(&self) -> Float { self.get_y_image() as Float}
    fn get_x_image(&self) -> usize {
        self.x.trunc() as usize
    }
    fn get_y_image(&self) -> usize {
        self.y.trunc() as usize
    }
    fn get_closest_sigma_level(&self) -> usize {
        self.sigma_level.trunc() as usize
    }
    fn apply_normalisation(&self, _: &na::Matrix3<Float>, _: Float) -> Self {
        panic!("TODO: SiftFeature apply_normalisation")
    }
    //TODO
    fn get_landmark_id(&self) -> Option<usize> {
        None
    }
    //TODO
    fn copy_with_landmark_id(&self, landmark_id: Option<usize>) -> Self {
        self.clone()
    }  
}

impl fmt::Display for SiftFeature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}, s: {}", self.x, self.y, self.sigma_level)
    }
}

impl PartialEq for SiftFeature {
    fn eq(&self, other: &Self) -> bool {
        self.get_location() == other.get_location()
    }
}

impl Eq for SiftFeature {}

impl Hash for SiftFeature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x = self.get_x_image();
        let y = self.get_y_image();
        (x,y).hash(state);
    }
}

//TODO: check these
impl SiftFeature {

    pub fn get_x(&self) -> Float{
        self.x
    }

    pub fn get_y(&self) -> Float{
        self.y
    }

    pub fn get_sigma_level(&self) -> Float{
        self.sigma_level
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
                extrema_vec.push(SiftFeature{x: x as Float,y: y as Float,sigma_level: sigma_level as Float, id: None});
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

pub fn sift_feature_refinement(extrema: &Vec<SiftFeature>, source_octave: &SiftOctave, runtime_params: &SiftRuntimeParams) -> Vec<SiftFeature> {
    extrema.iter().cloned().map(|x| processing::subpixel_refinement(source_octave, &x)).filter(|x| x.0 >= runtime_params.contrast_r).map(|x| x.1).filter(|x| hessian_response::reject_edge_response_filter(&source_octave.difference_of_gaussians,&source_octave.dog_x_gradient, x, runtime_params.edge_r)).collect()
    //extrema.iter().cloned().filter(|x| accept_edge_response_filter(source_octave, &x, runtime_params.edge_r)).collect()
    //extrema.clone()
}





