extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::Pyramid;
use sift::extrema;
use sift::image::Image;
use sift::image::{kernel::Kernel,laplace_kernel::LaplaceKernel,prewitt_kernel::PrewittKernel};
use sift::descriptor::orientation_histogram::generate_keypoints_from_extrema;
use sift::descriptor::local_image_descriptor::{is_rotated_keypoint_within_image,LocalImageDescriptor};
use sift::descriptor::feature_vector::FeatureVector;
use sift::KeyPoint;
use sift::visualize::display_histogram;

fn main() {
    let image_name = "circles";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let converted_file_out_path = format!("{}{}_histogram.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();

    
    let pyramid = Pyramid::build_pyramid(&gray_image, 3, 4, 0.5);
    let octave_level = 1;
    let sigma_level = 1;
    let octave = &pyramid.octaves[octave_level];

    let x_step = 1;
    let y_step = 1;
    let kernel_half_repeat = 1;
    let first_order_derivative_filter = PrewittKernel::new(kernel_half_repeat);
    let second_order_derivative_filter = LaplaceKernel::new(kernel_half_repeat);

    let features = extrema::detect_extrema(octave,sigma_level,first_order_derivative_filter.half_width(),first_order_derivative_filter.half_repeat(),x_step, y_step);
    let refined_features = extrema::extrema_refinement(&features, octave, &first_order_derivative_filter,&second_order_derivative_filter);
    let keypoints = refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x)).flatten().collect::<Vec<KeyPoint>>();
    let descriptors : Vec<LocalImageDescriptor> = keypoints.iter().filter(|x| is_rotated_keypoint_within_image(octave, x)).map(|x| LocalImageDescriptor::new(octave,x)).collect::<Vec<LocalImageDescriptor>>();
    let feature_vectors = descriptors.iter().map(|x| FeatureVector::new(x,octave_level)).collect::<Vec<FeatureVector>>();

    let new_image = display_histogram(&descriptors[0].descriptor_vector[1], 20, 400);

    new_image.to_image().save(converted_file_out_path).unwrap();



}