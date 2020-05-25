extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::image::Image;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let gray_image_path = format!("output/{}_gray_scale.{}",image_name,image_format);
    let converted_file_out_path = format!("output/{}_square.{}",image_name,image_format);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    image.save(gray_image_path).unwrap();


    let mut frame = Image::from_gray_image(&image);
    Image::draw_square(&mut frame, 50, 50, 10);
    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();
}