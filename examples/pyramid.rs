extern crate image as image_rs;
extern crate sift;

use std::path::Path;

use sift::pyramid::Pyramid;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    
    let pyramid = Pyramid::build_pyramid(&gray_image, 3, 3, 0.5);

    let first_octave = &pyramid.octaves[2];
    let first_ocatve_images = &first_octave.images;

    for i in 0..first_ocatve_images.len() {
        let image = &first_ocatve_images[i];
        let gray_image  = image.to_image();
        let name = format!("image_{}",i);
        let file_path = format!("{}{}.{}",image_out_folder,name,image_format);
        gray_image.save(file_path).unwrap();
    }

}