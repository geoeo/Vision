extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::pyramid::sift::{build_sift_pyramid,feature_vectors_from_pyramid, generate_match_pairs, sift_runtime_params::SiftRuntimeParams};
use vision::image::Image;

fn main() {
    
    let image_name = "blur";
    let image_name_2 = "blur_rotated";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let converted_file_out_path = format!("{}{}_matches.{}",image_out_folder,image_name,image_format);

    let runtime_params = SiftRuntimeParams {
        min_image_dimensions: (25,25),
        blur_half_factor: 6.0,
        orientation_histogram_window_factor: 1.0, //TODO: investigate
        edge_r: 10.0,
        contrast_r: 0.03,
        sigma_initial: 1.0,
        sigma_in: 0.5,
        octave_count: 4,
        sigma_count: 4
    };


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma();
    let display = Image::from_gray_image(&gray_image, false);
    let display_2 = Image::from_gray_image(&gray_image_2, false);
    
    let pyramid = build_sift_pyramid(&gray_image, &runtime_params);
    let pyramid_2 = build_sift_pyramid(&gray_image_2, &runtime_params);

    let all_features = feature_vectors_from_pyramid(&pyramid, &runtime_params);
    let all_features_2 = feature_vectors_from_pyramid(&pyramid_2, &runtime_params);

    let match_pairs = generate_match_pairs(&all_features, &all_features_2);

    println!("number of matched pairs: {}", match_pairs.len());

    //TODO: move to new mathcing functions
    //let match_dispay = display_matches(&display, &display_2, &all_features, &all_features_2, &match_pairs);

    //let new_image = match_dispay.to_image();
    //new_image.save(converted_file_out_path).unwrap();



}