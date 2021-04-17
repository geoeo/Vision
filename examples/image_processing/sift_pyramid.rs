extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::image::Image;
use vision::pyramid::sift::{build_sift_pyramid, sift_runtime_params::SiftRuntimeParams};


fn main() {
    let image_name = "circles";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma8();
    let image = Image::from_gray_image(&gray_image, false, false);
    

    let runtime_params = SiftRuntimeParams {
        min_image_dimensions: (25,25),
        blur_half_factor: 6.0,
        orientation_histogram_window_factor: 1.0, //TODO: investigate
        edge_r: 10.0,
        contrast_r: 0.03,
        sigma_initial: 1.0,
        sigma_in: 0.5,
        octave_count: 4,
        sigma_count: 4
    };

    
    let pyramid = build_sift_pyramid(image, &runtime_params);

    let first_octave = &pyramid.octaves[0];
    let ocatve_images = &first_octave.difference_of_gaussians;

    for i in 0..ocatve_images.len() {
        let image = &ocatve_images[i];
        let gray_image  = image.to_image();
        let name = format!("image_{}",i);
        let file_path = format!("{}{}.{}",image_out_folder,name,image_format);
        gray_image.save(file_path).unwrap();
    }

}