extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::Pyramid;
use sift::keypoint;
use sift::image::Image;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let converted_file_out_path = format!("{}{}_squares.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut display = Image::from_gray_image(&gray_image);
    
    let pyramid = Pyramid::build_pyramid(&gray_image, 3, 3, 0.5);

    let first_octave = &pyramid.octaves[0];
    let difference_of_gaussians = &first_octave.difference_of_gaussians;

    let features = keypoint::detect_extrema(&difference_of_gaussians[1], &difference_of_gaussians[0], &difference_of_gaussians[2], 1, 1);
    let number_of_features = features.len();
    let rows = first_octave.images[0].buffer.nrows();
    let cols = first_octave.images[0].buffer.ncols();
    let size = rows*cols;
    let percentage = number_of_features as f32/size as f32;

    println!("Features: {} out of {}, ({}%)",number_of_features, size, percentage);

    for feature in features {
        let x = feature.0;
        let y = feature.1;

        assert!(x < display.buffer.ncols());
        assert!(y < display.buffer.nrows());

        Image::draw_square(&mut display, x, y, 1);
    }

    let new_image = display.to_image();
    new_image.save(converted_file_out_path).unwrap();





}