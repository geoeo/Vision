extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::{Pyramid, runtime_params::RuntimeParams};
use sift::image::Image;
use sift::{keypoints_from_pyramid,keypoints_from_octave};
use sift::visualize::visualize_keypoint;

fn main() {
    
    //let image_name = "blur_rotated";
    //let image_name = "blur";
    let image_name = "circles";
    //let image_name = "beaver_90";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);

    let converted_file_out_path = format!("{}{}_keypoints_all.{}",image_out_folder,image_name,image_format);

    println!("Processing Image: {}", image_name);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut display = Image::from_gray_image(&gray_image, false);
    
    //TODO: move inital blur params here
    let runtime_params = RuntimeParams {
        blur_half_factor: 4.0, //TODO: lowering <= 4 this causes algorithm to become unstable
        orientation_histogram_window_factor: 1, //TODO: investigate
        edge_r: 10.0,
        contrast_r: 0.03,
        sigma_initial: 1.6,
        octave_count: 8,
        sigma_count: 3
    };

    //TODO: experiment with blur half width and pyramid params
    let pyramid = Pyramid::build_pyramid(&gray_image,&runtime_params);

    let all_keypoints = keypoints_from_pyramid(&pyramid, &runtime_params);
    //let all_keypoints = keypoints_from_octave(&pyramid,2, &runtime_params);

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