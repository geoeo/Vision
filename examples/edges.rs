extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::image::Image;
use vision::filter::{prewitt_kernel::PrewittKernel,filter_1d_convolution};
use vision::GradientDirection;

fn main() {
    let image_name = "squares";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let blur_hor_file_out_path = format!("output/{}_hor_edge.{}",image_name,image_format);
    let blur_vert_file_out_path = format!("output/{}_vert_edge.{}",image_name,image_format);


    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let frame = Image::from_gray_image(&image, false, false);
    let vec = vec![&frame];

    let filter_kernel = PrewittKernel::new();
    
    let blur_hor = filter_1d_convolution(&vec,0,GradientDirection::HORIZINTAL, &filter_kernel, false);

    let blur_hor_image = blur_hor.to_image();
    blur_hor_image.save(blur_hor_file_out_path).unwrap();

    let blur_vert = filter_1d_convolution(&vec,0,GradientDirection::VERTICAL, &filter_kernel, false);
    let blur_vert_image = blur_vert.to_image();
    blur_vert_image.save(blur_vert_file_out_path).unwrap();

}