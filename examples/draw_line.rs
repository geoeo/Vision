extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::image::Image;
use vision::visualize::{draw_line,draw_circle};
use vision::float;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let gray_image_path = format!("output/{}_gray_scale.{}",image_name,image_format);
    let converted_file_out_path = format!("output/{}_line.{}",image_name,image_format);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    image.save(gray_image_path).unwrap();


    
    let angle_degrees = 120.0;
    let angle_rad = angle_degrees*float::consts::PI/180.0;
    let mut frame = Image::from_gray_image(&image, false);
    draw_line(&mut frame, 100, 50, 20.0,angle_rad);
    draw_circle(&mut frame, 100, 50, 3.0);
    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();
}