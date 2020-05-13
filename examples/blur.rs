extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::image::{Image,filter, image_encoding::ImageEncoding};

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let converted_file_out_path = format!("output/{}_blur.{}",image_name,image_format);


    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let frame = Image::from_gray_image(&image);
    let mut target = Image::empty(frame.buffer.nrows(), frame.buffer.ncols(), ImageEncoding::U8);

    filter::gaussian_1_d_convolution_horizontal(&frame,&mut target, 0.0, 5.5);
    filter::gaussian_1_d_convolution_vertical(&frame,&mut target, 0.0, 5.5);

    let new_image = target.to_image();

    new_image.save(converted_file_out_path).unwrap();
}