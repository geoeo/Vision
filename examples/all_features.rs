extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::pyramid::sift::{build_sift_pyramid, sift_runtime_params::SiftRuntimeParams,feature_vectors_from_pyramid};
use vision::image::Image;
use vision::reconstruct_original_coordiantes;
use vision::visualize::draw_square;

fn main() {
    
    let image_name = "blur";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);

    let converted_file_out_path = format!("{}{}_squares_all.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let image = Image::from_gray_image(&gray_image, false);
    let mut display = Image::from_gray_image(&gray_image, false);

    // let runtime_params = RuntimeParams {
    //     blur_half_width: 9,
    //     orientation_histogram_window_size: 9,
    //     edge_r: 2.5,
    //     contrast_r: 0.1,
    //     sigma_initial: 1.0,
    //     octave_count: 4,
    //     sigma_count: 4
    // };

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

    let all_features = feature_vectors_from_pyramid(&pyramid, &runtime_params);

    let number_of_features = all_features.len();

    let rows = pyramid.octaves[0].images[0].buffer.nrows();
    let cols = pyramid.octaves[0].images[0].buffer.ncols();
    let size = rows*cols;
    let percentage = number_of_features as f32/size as f32;

    println!("Features from Image: {} out of {}, ({}%)",number_of_features, size, percentage);

    for feature in all_features {
        let(x,y) = reconstruct_original_coordiantes(feature.x, feature.y,feature.octave_level as u32);

        assert!(x < display.buffer.ncols());
        assert!(y < display.buffer.nrows());

        draw_square(&mut display, x, y, 1);
    }


    let new_image = display.to_image();

    new_image.save(converted_file_out_path).unwrap();



}