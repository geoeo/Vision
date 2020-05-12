extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::image::{Image,filter};

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let converted_file_out_path = format!("output/{}_blur.{}",image_name,image_format);


    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut frame = Image::from_gray_image(&image);
    filter::gaussian_1_d_convolution_horizontal(&mut frame, 0.0, 1.0);
    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();
}