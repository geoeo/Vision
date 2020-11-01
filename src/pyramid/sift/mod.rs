extern crate image as image_rs;

use image_rs::GrayImage;

use crate::Float;
use crate::image::Image;
use crate::filter::{gauss_kernel::GaussKernel1D,gaussian_2_d_convolution};
use crate::pyramid::{Pyramid,sift::{sift_octave::SiftOctave,sift_runtime_params::SiftRuntimeParams}};
use crate::features::sift_feature;
use crate::matching::sift_descriptor::{
    feature_vector::FeatureVector,
    orientation_histogram::generate_keypoints_from_extrema,
    local_image_descriptor::{is_rotated_keypoint_within_image,LocalImageDescriptor},
    keypoint::KeyPoint
};


pub mod sift_runtime_params;
pub mod sift_octave;

type SiftPyramid = Pyramid<SiftOctave>;
pub const RELATIVE_MATCH_THRESHOLD: Float = 0.6;

pub fn build_sift_pyramid(base_gray_image: &GrayImage, runtime_params: &SiftRuntimeParams) -> SiftPyramid {
    let mut octaves: Vec<SiftOctave> = Vec::with_capacity(runtime_params.octave_count);

    let base_image = Image::from_gray_image(base_gray_image, false);
    let upsample = Image::upsample_double(&base_image, false);

    //TODO: check this
    let blur_width = SiftOctave::generate_blur_radius(runtime_params.blur_half_factor, runtime_params.sigma_in);
    let kernel = GaussKernel1D::new(0.0, runtime_params.sigma_in,1,blur_width);
    let initial_blur =  gaussian_2_d_convolution(&upsample, &kernel, false);
    
    let mut octave_image = initial_blur;
    let mut sigma = runtime_params.sigma_initial;
    let sigma_count = runtime_params.sigma_count;


    for i in 0..runtime_params.octave_count {

        if i > 0 {
            octave_image = Image::downsample_half(&octaves[i-1].images[runtime_params.sigma_count], false);
            sigma = octaves[i-1].sigmas[sigma_count];
        }

        let new_octave = SiftOctave::build_octave(&octave_image, runtime_params.sigma_count, sigma, runtime_params);

        octaves.push(new_octave);
        

    }

    Pyramid{octaves}
}

pub fn feature_vectors_from_pyramid(pyramid: &Pyramid<SiftOctave>, runtime_params:&SiftRuntimeParams) -> Vec<FeatureVector> {

    let mut all_vectors = Vec::<Vec<FeatureVector>>::new();

    for octave_level in 0..pyramid.octaves.len() {
        for sigma_level in 1..pyramid.octaves[0].s()+1 {
            all_vectors.push(feature_vectors_from_octave(pyramid,octave_level,sigma_level,runtime_params));
        }
    }

    all_vectors.into_iter().flatten().collect()

}

pub fn keypoints_from_pyramid(pyramid: &Pyramid<SiftOctave>, runtime_params:&SiftRuntimeParams) -> Vec<KeyPoint> {

    let mut all_vectors = Vec::<Vec<KeyPoint>>::new();

    for octave_level in 0..pyramid.octaves.len() {
        all_vectors.push(keypoints_from_octave(pyramid, octave_level,runtime_params));
    }

    all_vectors.into_iter().flatten().collect()

}

pub fn feature_vectors_from_octave(pyramid: &Pyramid<SiftOctave>, octave_level: usize, sigma_level: usize, runtime_params:&SiftRuntimeParams) -> Vec<FeatureVector> {
    let x_step = 1;
    let y_step = 1;

    let octave = &pyramid.octaves[octave_level];

    let features = sift_feature::detect_sift_feature(octave,sigma_level,x_step, y_step);
    let refined_features = sift_feature::sift_feature_refinement(&features, octave, runtime_params);
    let keypoints = refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x, runtime_params)).flatten().collect::<Vec<KeyPoint>>();
    let descriptors = keypoints.iter().filter(|x| is_rotated_keypoint_within_image(octave, x)).map(|x| LocalImageDescriptor::new(octave,x)).collect::<Vec<LocalImageDescriptor>>();
    descriptors.iter().map(|x| FeatureVector::new(x,octave_level)).collect::<Vec<FeatureVector>>()
}

pub fn keypoints_from_octave(pyramid: &Pyramid<SiftOctave>, octave_level: usize, runtime_params: &SiftRuntimeParams) -> Vec<KeyPoint> {
    let mut all_vectors = Vec::<Vec<KeyPoint>>::new();

    for dog_level in 1..pyramid.octaves[0].s()+1 {
        all_vectors.push(keypoints_from_sigma(pyramid, octave_level,dog_level, runtime_params));
    }

    all_vectors.into_iter().flatten().collect()
}

pub fn keypoints_from_sigma(pyramid: &Pyramid<SiftOctave>, octave_level: usize, dog_level: usize, runtime_params: &SiftRuntimeParams) -> Vec<KeyPoint> {
    let x_step = 1;
    let y_step = 1;

    let octave = &pyramid.octaves[octave_level];

    let features = sift_feature::detect_sift_feature(octave,dog_level,x_step, y_step);
    let refined_features = sift_feature::sift_feature_refinement(&features, octave, runtime_params);
    refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x, runtime_params)).flatten().filter(|x| is_rotated_keypoint_within_image(octave, x)).collect::<Vec<KeyPoint>>()
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


