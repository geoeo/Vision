
use self::pyramid::Pyramid;
use self::descriptor::feature_vector::FeatureVector;
use self::descriptor::orientation_histogram::generate_keypoints_from_extrema;
use self::descriptor::local_image_descriptor::{is_rotated_keypoint_within_image,LocalImageDescriptor};
use self::image::{kernel::Kernel,prewitt_kernel::PrewittKernel};
use std::fmt;

pub mod image;
pub mod pyramid;
pub mod extrema;
pub mod descriptor;
pub mod visualize;

macro_rules! define_float {
    ($f:tt) => {
        pub use std::$f as float;
        pub type Float = $f;
    }
}

pub const RELATIVE_MATCH_THRESHOLD: Float = 0.6;
pub const BLUR_HALF_WIDTH: usize = 8; // TODO: make this a input param to pyramid
pub const ORIENTATION_HISTOGRAM_WINDOW_SIZE: usize = 8; // TODO: make this a input param to pyramid

define_float!(f64);

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize
} 

impl fmt::Display for ExtremaParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}, s: {}", self.x, self.y, self.sigma_level)
    }
}

#[derive(Debug,Clone)]
pub struct KeyPoint {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize,
    pub octave_level: usize,
    pub orientation: Float
    //TODO: maybe put octave/orientation histogram here as well for debugging
} 

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum GradientDirection {
    HORIZINTAL,
    VERTICAL,
    SIGMA
}

pub fn feature_vectors_from_pyramid(pyramid: &Pyramid) -> Vec<FeatureVector> {

    let mut all_vectors = Vec::<Vec<FeatureVector>>::new();

    for octave_level in 0..pyramid.octaves.len() {
        let octave = &pyramid.octaves[octave_level];
        for sigma_level in 1..octave.sigmas.len()-2 {
            all_vectors.push(feature_vectors_from_octave(pyramid,octave_level,sigma_level));
        }
    }

    all_vectors.into_iter().flatten().collect()

}

pub fn keypoints_from_pyramid(pyramid: &Pyramid) -> Vec<KeyPoint> {

    let mut all_vectors = Vec::<Vec<KeyPoint>>::new();

    for octave_level in 0..pyramid.octaves.len() {
        all_vectors.push(keypoints_from_octave(pyramid, octave_level));
    }

    all_vectors.into_iter().flatten().collect()

}


pub fn feature_vectors_from_octave(pyramid: &Pyramid, octave_level: usize, sigma_level: usize) -> Vec<FeatureVector> {
    let x_step = 1;
    let y_step = 1;
    let first_order_derivative_filter = PrewittKernel::new();

    let octave = &pyramid.octaves[octave_level];

    let features = extrema::detect_extrema(octave,sigma_level,first_order_derivative_filter.half_width(),first_order_derivative_filter.half_repeat(),x_step, y_step);
    let refined_features = extrema::extrema_refinement(&features, octave, &first_order_derivative_filter);
    let keypoints = refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x)).flatten().collect::<Vec<KeyPoint>>();
    let descriptors = keypoints.iter().filter(|x| is_rotated_keypoint_within_image(octave, x)).map(|x| LocalImageDescriptor::new(octave,x)).collect::<Vec<LocalImageDescriptor>>();
    descriptors.iter().map(|x| FeatureVector::new(x,octave_level)).collect::<Vec<FeatureVector>>()
}

pub fn keypoints_from_octave(pyramid: &Pyramid, octave_level: usize) -> Vec<KeyPoint> {
    let mut all_vectors = Vec::<Vec<KeyPoint>>::new();

    let octave = &pyramid.octaves[octave_level];
    for dog_level in 1..octave.difference_of_gaussians.len()-1 {
        all_vectors.push(keypoints_from_sigma(pyramid,octave_level,dog_level));
    }

    all_vectors.into_iter().flatten().collect()
}

pub fn keypoints_from_sigma(pyramid: &Pyramid, octave_level: usize, dog_level: usize) -> Vec<KeyPoint> {
    let x_step = 1;
    let y_step = 1;
    let first_order_derivative_filter = PrewittKernel::new();

    let octave = &pyramid.octaves[octave_level];

    let features = extrema::detect_extrema(octave,dog_level,first_order_derivative_filter.half_width(),first_order_derivative_filter.half_repeat(),x_step, y_step);
    let refined_features = extrema::extrema_refinement(&features, octave, &first_order_derivative_filter);
    refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x)).flatten().collect::<Vec<KeyPoint>>()
}

pub fn reconstruct_original_coordiantes(x: usize, y: usize, octave_level: u32) -> (usize,usize) {
    let factor = 2usize.pow(octave_level);
    (x*factor,y*factor)
}


pub fn match_feature(a: &FeatureVector, bs: &Vec<FeatureVector>) -> Option<usize> { 
    assert!(bs.len() > 1);

    //TODO: distance seems buggy 
    let mut index_distances = bs.iter().enumerate().map(|b| (b.0,a.distance_between(b.1))).collect::<Vec<(usize,Float)>>();
    index_distances.sort_by(|x,y| x.1.partial_cmp(&y.1).unwrap());

    let nearest_distance = index_distances[0].1;
    let second_nearest_distance = index_distances[1].1;

    let nearest_index = index_distances[0].0;

    match nearest_distance < RELATIVE_MATCH_THRESHOLD*second_nearest_distance {
        true => Some(nearest_index),
        false => None
    }
}

pub fn generate_match_pairs(feature_list_a: &Vec<FeatureVector>, feature_list_b: &Vec<FeatureVector>) -> Vec<(usize,usize)> {
    feature_list_a.iter().enumerate().map(|a| (a.0,match_feature(a.1,feature_list_b))).filter(|&x| x.1 != None).map(|x| (x.0,x.1.unwrap())).collect::<Vec<(usize,usize)>>()
}

pub fn round(number: Float, dp: i32) -> Float {
    let n = (10.0 as Float).powi(dp);
    (number * n).round()/n
}