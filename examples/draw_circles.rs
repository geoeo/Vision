extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::image::Image;
use vision::visualize::{draw_circle,draw_points};
use vision::features::geometry::shape::circle::circle_bresenham;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let gray_image_path = format!("output/{}_gray_scale.{}",image_name,image_format);
    let converted_file_out_path = format!("output/{}_circle.{}",image_name,image_format);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    image.save(gray_image_path).unwrap();


    let mut frame = Image::from_gray_image(&image, false);
    draw_circle(&mut frame, 100, 50, 3.0, 64.0);
    let circle = circle_bresenham(35, 35, 5);
    draw_points(&mut frame, &circle.shape.get_points(), 64.0);
    draw_points(&mut frame, &circle.get, 64.0);
    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();
}