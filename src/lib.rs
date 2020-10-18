
use self::pyramid::{sift_octave::SiftOctave,Pyramid,runtime_params::RuntimeParams};
use self::sift_descriptor::{
    feature_vector::FeatureVector,
    orientation_histogram::generate_keypoints_from_extrema,
    local_image_descriptor::{is_rotated_keypoint_within_image,LocalImageDescriptor},
    keypoint::KeyPoint
};

pub mod filter;
pub mod image;
pub mod pyramid;
pub mod feature;
pub mod sift_descriptor;
pub mod visualize;
pub mod vo;

macro_rules! define_float {
    ($f:tt) => {
        pub use std::$f as float;
        pub type Float = $f;
    }
}

pub const RELATIVE_MATCH_THRESHOLD: Float = 0.6;
//pub const BLUR_HALF_WIDTH: usize = 9; // TODO: make this a input param to pyramid / Scale this with octave level
//pub const ORIENTATION_HISTOGRAM_WINDOW_SIZE: usize = 9; // TODO: make this a input param to pyramid
//pub const EDGE_R: Float = 2.5; // TODO: make this a input param to pyramid
//pub const CONTRAST_R: Float = 0.1; // TODO: make this a input param to pyramid

define_float!(f64);

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum GradientDirection {
    HORIZINTAL,
    VERTICAL,
    SIGMA
}

pub fn feature_vectors_from_pyramid(pyramid: &Pyramid<SiftOctave>, runtime_params:&RuntimeParams) -> Vec<FeatureVector> {

    let mut all_vectors = Vec::<Vec<FeatureVector>>::new();

    for octave_level in 0..pyramid.octaves.len() {
        let octave = &pyramid.octaves[octave_level];
        for sigma_level in 1..pyramid.s+1 {
            all_vectors.push(feature_vectors_from_octave(pyramid,octave_level,sigma_level,runtime_params));
        }
    }

    all_vectors.into_iter().flatten().collect()

}

pub fn keypoints_from_pyramid(pyramid: &Pyramid<SiftOctave>, runtime_params:&RuntimeParams) -> Vec<KeyPoint> {

    let mut all_vectors = Vec::<Vec<KeyPoint>>::new();

    for octave_level in 0..pyramid.octaves.len() {
        all_vectors.push(keypoints_from_octave(pyramid, octave_level,runtime_params));
    }

    all_vectors.into_iter().flatten().collect()

}

//TODO: unify these methods
pub fn feature_vectors_from_octave(pyramid: &Pyramid<SiftOctave>, octave_level: usize, sigma_level: usize, runtime_params:&RuntimeParams) -> Vec<FeatureVector> {
    let x_step = 1;
    let y_step = 1;

    let octave = &pyramid.octaves[octave_level];

    let features = feature::detect_sift_feature(octave,sigma_level,x_step, y_step);
    let refined_features = feature::extrema_refinement(&features, octave,octave_level, runtime_params);
    let keypoints = refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x, runtime_params)).flatten().collect::<Vec<KeyPoint>>();
    let descriptors = keypoints.iter().filter(|x| is_rotated_keypoint_within_image(octave, x)).map(|x| LocalImageDescriptor::new(octave,x)).collect::<Vec<LocalImageDescriptor>>();
    descriptors.iter().map(|x| FeatureVector::new(x,octave_level)).collect::<Vec<FeatureVector>>()
}

pub fn keypoints_from_octave(pyramid: &Pyramid<SiftOctave>, octave_level: usize, runtime_params: &RuntimeParams) -> Vec<KeyPoint> {
    let mut all_vectors = Vec::<Vec<KeyPoint>>::new();

    for dog_level in 1..pyramid.s+1 {
        all_vectors.push(keypoints_from_sigma(pyramid, octave_level,dog_level, runtime_params));
    }

    all_vectors.into_iter().flatten().collect()
}

pub fn keypoints_from_sigma(pyramid: &Pyramid<SiftOctave>, octave_level: usize, dog_level: usize, runtime_params: &RuntimeParams) -> Vec<KeyPoint> {
    let x_step = 1;
    let y_step = 1;

    let octave = &pyramid.octaves[octave_level];

    let features = feature::detect_sift_feature(octave,dog_level,x_step, y_step);
    let refined_features = feature::extrema_refinement(&features, octave,octave_level, runtime_params);
    refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x, runtime_params)).flatten().filter(|x| is_rotated_keypoint_within_image(octave, x)).collect::<Vec<KeyPoint>>()
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