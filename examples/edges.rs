extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::image::{Image,filter, filter::FilterDirection2D, prewitt_kernel::PrewittKernel};

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let blur_hor_file_out_path = format!("output/{}_hor_edge.{}",image_name,image_format);
    let blur_vert_file_out_path = format!("output/{}_vert_edge.{}",image_name,image_format);


    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let frame = Image::from_gray_image(&image, false);

    let filter_kernel = PrewittKernel::new(1);
    
    let blur_hor = filter::filter_1d_convolution(&frame,FilterDirection2D::HORIZINTAL, &filter_kernel);

    let blur_hor_image = blur_hor.to_image();
    blur_hor_image.save(blur_hor_file_out_path).unwrap();

    let blur_vert = filter::filter_1d_convolution(&frame,FilterDirection2D::VERTICAL, &filter_kernel);
    let blur_vert_image = blur_vert.to_image();
    blur_vert_image.save(blur_vert_file_out_path).unwrap();

}