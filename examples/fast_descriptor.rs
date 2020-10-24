extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::image::Image;
use sift::visualize::{draw_circle,draw_points};
use sift::features::geometry::{Geometry,circle::circle_bresenham, point::Point};
use sift::fast_descriptor::FastDescriptor;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let gray_image_path = format!("output/{}_gray_scale.{}",image_name,image_format);
    let converted_file_out_path = format!("output/{}_fast.{}",image_name,image_format);

    println!("Processing Image: {}", image_name);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut frame = Image::from_gray_image(&image, false);

    let circle = circle_bresenham(35, 35, 3);
    // for offset in &circle.geometry.offsets {
    //     println!("{:?}",offset);
    // }

    //let fast_descriptor = FastDescriptor::from_circle(&circle);
    //FastDescriptor::print_continuous_offsets(&fast_descriptor);

    let valid_descriptors = FastDescriptor::compute_valid_descriptors(&frame,3,0.2,12,(10,10));
    for (valid_descriptor,i) in valid_descriptors {
        let slice = valid_descriptor.get_wrapping_slice(i, 12);
        let points = Geometry::points(valid_descriptor.x_center, valid_descriptor.y_center, &slice);
        draw_points(&mut frame, &points, 64.0);
    }




    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();

}