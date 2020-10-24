extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::image::Image;
use sift::visualize::{draw_circle,draw_points};
use sift::features::geometry::circle::circle_bresenham;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let gray_image_path = format!("output/{}_gray_scale.{}",image_name,image_format);
    let converted_file_out_path = format!("output/{}_circle.{}",image_name,image_format);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    image.save(gray_image_path).unwrap();


    let mut frame = Image::from_gray_image(&image, false);
    draw_circle(&mut frame, 100, 50, 3.0);
    let circle = circle_bresenham(35, 35, 5);
    draw_points(&mut frame, &circle.geometry.get_points(), 64.0);
    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();
}