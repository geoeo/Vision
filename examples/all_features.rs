extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::Pyramid;
use sift::image::Image;
use sift::{feature_vectors_from_octave,reconstruct_original_coordiantes,feature_vectors_from_pyramid};

fn main() {
    
    let image_name = "beaver_90";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);

    let converted_file_out_path = format!("{}{}_squares_all.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut display = Image::from_gray_image(&gray_image, false);
    
    let pyramid = Pyramid::build_pyramid(&gray_image, 3, 3, 0.5);

    let all_features = feature_vectors_from_pyramid(&pyramid);


    let number_of_features = all_features.len();

    let rows = pyramid.octaves[0].images[0].buffer.nrows();
    let cols = pyramid.octaves[0].images[0].buffer.ncols();
    let size = rows*cols;
    let percentage = number_of_features as f32/size as f32;

    println!("Features from Image: {} out of {}, ({}%)",number_of_features, size, percentage);

    for feature in all_features {
        let(x,y) = reconstruct_original_coordiantes(feature.x, feature.y,feature.octave_level as u32);

        assert!(x < display.buffer.ncols());
        assert!(y < display.buffer.nrows());

        Image::draw_square(&mut display, x, y, 1);
    }


    let new_image = display.to_image();

    new_image.save(converted_file_out_path).unwrap();





}