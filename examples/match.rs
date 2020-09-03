extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::Pyramid;
use sift::image::Image;
use sift::{feature_vectors_from_octave,reconstruct_original_coordiantes,feature_vectors_from_pyramid, generate_match_pairs};

fn main() {
    
    let image_name = "lenna_90";
    let image_name_2 = "lenna";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let converted_file_out_path = format!("{}{}_matches.{}",image_out_folder,image_name,image_format);



    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma();
    let display = Image::from_gray_image(&gray_image, false);
    let display_2 = Image::from_gray_image(&gray_image_2, false);
    
    let pyramid = Pyramid::build_pyramid(&gray_image, 3, 3, 0.5);
    let pyramid_2 = Pyramid::build_pyramid(&gray_image_2, 3, 3, 0.5);

    let all_features = feature_vectors_from_pyramid(&pyramid);
    let all_features_2 = feature_vectors_from_pyramid(&pyramid_2);

    let match_pairs = generate_match_pairs(&all_features, &all_features_2);

    println!("number of matched pairs: {}", match_pairs.len());

    let match_dispay = Image::display_matches(&display, &display_2, &all_features, &all_features_2, &match_pairs);

    let new_image = match_dispay.to_image();
    new_image.save(converted_file_out_path).unwrap();



}