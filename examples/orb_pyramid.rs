extern crate image as image_rs;
extern crate sift;

use std::path::Path;
use sift::pyramid::orb::{build_orb_pyramid,generate_features_for_pyramid,generate_descriptors_for_pyramid,  orb_runtime_parameters::OrbRuntimeParameters};

fn main() {
    let image_name = "circles";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();

    let runtime_params = OrbRuntimeParameters {
        min_image_dimensions: (50,50),
        sigma: 0.8,
        blur_radius: 5.0,
        octave_count: 4,
        harris_k: 0.04,
        fast_circle_radius: 3,
        fast_threshold_factor: 0.2,
        fast_consecutive_pixels: 12,
        fast_grid_size: (10,10),
        brief_n: 256,
        brief_s: 31
    };
    
    let pyramid = build_orb_pyramid(&gray_image, &runtime_params);
    let feature_pyramid = generate_features_for_pyramid(&pyramid, &runtime_params);
    let feautre_descriptors = generate_descriptors_for_pyramid(&pyramid,&feature_pyramid,&runtime_params);

    // for i in 0..pyramid.octaves.len() {
    //     let octave = &pyramid.octaves[i];
    //     let image = &octave.images[0];
    //     let gray_image  = image.to_image();

    //     let name = format!("orb_image_{}",i);
    //     let file_path = format!("{}{}.{}",image_out_folder,name,image_format);
    //     gray_image.save(file_path).unwrap();
    // }

}