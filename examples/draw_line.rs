extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::image::Image;
use vision::visualize::{draw_line,draw_circle, draw_points};
use vision::float;
use vision::features::geometry::{line::line_bresenham,point::Point};

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let gray_image_path = format!("output/{}_gray_scale.{}",image_name,image_format);
    let converted_file_out_path = format!("output/{}_line.{}",image_name,image_format);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma8();
    image.save(gray_image_path).unwrap();


    
    let angle_degrees = 120.0;
    let angle_rad = angle_degrees*float::consts::PI/180.0;
    let mut frame = Image::from_gray_image(&image, false, false);
    //draw_line(&mut frame, 100, 50, 20.0,angle_rad);


    let line = line_bresenham(&Point::new(0, 0), &Point::new(10, 50));
    draw_points(&mut frame, &line.points, 64.0);

    // let line_2 = line_bresenham(&Point::new(100, 25), &Point::new(50, 50));
    // draw_points(&mut frame, &line_2.points, 64.0);

    // let line_3 = line_bresenham(&Point::new(100, 200), &Point::new(150, 10));
    // draw_points(&mut frame, &line_3.points, 64.0);

    let new_image = frame.to_image();
    new_image.save(converted_file_out_path).unwrap();
}