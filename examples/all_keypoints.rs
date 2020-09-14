extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::Pyramid;
use sift::image::Image;
use sift::{feature_vectors_from_octave,reconstruct_original_coordiantes,keypoints_from_pyramid, keypoints_from_octave};
use sift::visualize::visualize_keypoint;

fn main() {
    
    let image_name = "circles";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);

    let converted_file_out_path = format!("{}{}_keypoints_all.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut display = Image::from_gray_image(&gray_image, false);
    
    let pyramid = Pyramid::build_pyramid(&gray_image, 5, 3, 0.25);

    let all_keypoints = keypoints_from_pyramid(&pyramid);
    //let all_keypoints = keypoints_from_octave(&pyramid, 2);

    let number_of_features = all_keypoints.len();

    let rows = pyramid.octaves[0].images[0].buffer.nrows();
    let cols = pyramid.octaves[0].images[0].buffer.ncols();
    let size = rows*cols;
    let percentage = number_of_features as f32/size as f32;

    println!("Keypoints from Image: {} out of {}, ({}%)",number_of_features, size, percentage);

    for keypoint in all_keypoints {
        visualize_keypoint(&mut display, &keypoint);
    }


    let new_image = display.to_image();

    new_image.save(converted_file_out_path).unwrap();





}