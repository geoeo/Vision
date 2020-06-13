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
    let converted_refined_file_out_path = format!("{}{}_refined_squares.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut display = Image::from_gray_image(&gray_image, false);
    let mut refined_display = Image::from_gray_image(&gray_image, false);
    
    let pyramid = Pyramid::build_pyramid(&gray_image, 3, 3, 0.5);

    let first_octave = &pyramid.octaves[0];
    let kernel_half_width = 1;

    let features = keypoint::detect_extrema(first_octave,1,kernel_half_width,1, 1);
    let refined_features = keypoint::extrema_refinement(&features, first_octave, kernel_half_width);

    let number_of_features = features.len();
    let number_of_refined_features = refined_features.len();
    let rows = first_octave.images[0].buffer.nrows();
    let cols = first_octave.images[0].buffer.ncols();
    let size = rows*cols;
    let percentage = number_of_features as f32/size as f32;
    let refined_percentage = number_of_refined_features as f32/size as f32;

    println!("Features: {} out of {}, ({}%)",number_of_features, size, percentage);
    println!("Refined Features: {} out of {}, ({}%)",refined_percentage, size, refined_percentage);

    for feature in features {
        let x = feature.x;
        let y = feature.y;

        assert!(x < display.buffer.ncols());
        assert!(y < display.buffer.nrows());

        Image::draw_square(&mut display, x, y, 1);
    }

    for feature in refined_features {
        let x = feature.x;
        let y = feature.y;

        assert!(x < display.buffer.ncols());
        assert!(y < display.buffer.nrows());

        Image::draw_square(&mut refined_display, x, y, 1);
    }

    let new_image = display.to_image();
    let refined_new_image = refined_display.to_image();

    new_image.save(converted_file_out_path).unwrap();
    refined_new_image.save(converted_refined_file_out_path).unwrap();





}