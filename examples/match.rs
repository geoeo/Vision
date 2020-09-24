extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::{Pyramid, runtime_params::RuntimeParams};
use sift::image::Image;
use sift::{feature_vectors_from_octave,reconstruct_original_coordiantes,feature_vectors_from_pyramid, generate_match_pairs};
use sift::visualize::display_matches;

fn main() {
    
    let image_name = "beaver_90";
    let image_name_2 = "beaver";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let converted_file_out_path = format!("{}{}_matches.{}",image_out_folder,image_name,image_format);

    let runtime_params = RuntimeParams {
        blur_half_width: 9,
        orientation_histogram_window_size: 9,
        edge_r: 2.5,
        contrast_r: 0.1,
        sigma_initial: 1.0,
        octave_count: 4,
        sigma_count: 4
    };


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma();
    let display = Image::from_gray_image(&gray_image, false);
    let display_2 = Image::from_gray_image(&gray_image_2, false);
    
    let pyramid = Pyramid::build_pyramid(&gray_image, &runtime_params);
    let pyramid_2 = Pyramid::build_pyramid(&gray_image_2, &runtime_params);

    let all_features = feature_vectors_from_pyramid(&pyramid, &runtime_params);
    let all_features_2 = feature_vectors_from_pyramid(&pyramid_2, &runtime_params);

    let match_pairs = generate_match_pairs(&all_features, &all_features_2);

    println!("number of matched pairs: {}", match_pairs.len());

    let match_dispay = display_matches(&display, &display_2, &all_features, &all_features_2, &match_pairs);

    let new_image = match_dispay.to_image();
    new_image.save(converted_file_out_path).unwrap();



}