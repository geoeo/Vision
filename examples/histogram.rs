extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::{build_sift_pyramid, runtime_params::RuntimeParams};
use sift::feature;
use sift::filter::{kernel::Kernel,prewitt_kernel::PrewittKernel};
use sift::descriptor::orientation_histogram::generate_keypoints_from_extrema;
use sift::descriptor::local_image_descriptor::{is_rotated_keypoint_within_image,LocalImageDescriptor};
use sift::descriptor::feature_vector::FeatureVector;
use sift::descriptor::keypoint::KeyPoint;
use sift::visualize::display_histogram;

fn main() {
    let image_name = "circles";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let converted_file_out_path = format!("{}{}_histogram.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();

    let runtime_params = RuntimeParams {
        blur_half_factor: 6.0,
        orientation_histogram_window_factor: 1.0, //TODO: investigate
        edge_r: 10.0,
        contrast_r: 0.03,
        sigma_initial: 1.0,
        sigma_in: 0.5,
        octave_count: 4,
        sigma_count: 4
    };

    
    let pyramid = build_sift_pyramid(&gray_image, &runtime_params);
    let octave_level = 1;
    let sigma_level = 1;
    let octave = &pyramid.octaves[octave_level];

    let x_step = 1;
    let y_step = 1;
    let first_order_derivative_filter = PrewittKernel::new();

    let features = feature::detect_sift_feature(octave,sigma_level,x_step, y_step);
    let refined_features = feature::extrema_refinement(&features, octave,octave_level, &runtime_params);
    let keypoints = refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x, &runtime_params)).flatten().collect::<Vec<KeyPoint>>();
    let descriptors : Vec<LocalImageDescriptor> = keypoints.iter().filter(|x| is_rotated_keypoint_within_image(octave, x)).map(|x| LocalImageDescriptor::new(octave,x)).collect::<Vec<LocalImageDescriptor>>();
    let feature_vectors = descriptors.iter().map(|x| FeatureVector::new(x,octave_level)).collect::<Vec<FeatureVector>>();

    let new_image = display_histogram(&descriptors[0].descriptor_vector[1], 20, 400);

    new_image.to_image().save(converted_file_out_path).unwrap();



}