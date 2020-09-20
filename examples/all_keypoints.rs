extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::Pyramid;
use sift::image::Image;
use sift::keypoints_from_pyramid;
use sift::visualize::visualize_keypoint;

fn main() {
    
    let image_name = "circles";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);

    let converted_file_out_path = format!("{}{}_keypoints_all.{}",image_out_folder,image_name,image_format);

    println!("Processing Image: {}", image_name);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut display = Image::from_gray_image(&gray_image, false);
    
    //TODO: experiment with blur half width and pyramid params
    //let pyramid = Pyramid::build_pyramid(&gray_image, 2, 3, 0.5);
    let pyramid = Pyramid::build_pyramid(&gray_image, 3, 3, 1.5);

    let all_keypoints = keypoints_from_pyramid(&pyramid);
    //let all_keypoints = keypoints_from_octave(&pyramid, 1);

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