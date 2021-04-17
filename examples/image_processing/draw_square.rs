extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::image::Image;
use vision::visualize::draw_square;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let gray_image_path = format!("output/{}_gray_scale.{}",image_name,image_format);
    let converted_file_out_path = format!("output/{}_square.{}",image_name,image_format);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma8();
    image.save(gray_image_path).unwrap();


    let mut frame = Image::from_gray_image(&image, false, false);
    draw_square(&mut frame, 50, 50, 10);
    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();
}