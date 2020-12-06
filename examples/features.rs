extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::pyramid::sift::{build_sift_pyramid, sift_runtime_params::SiftRuntimeParams};
use vision::features::sift_feature;
use vision::image::Image;
use vision::matching::sift_descriptor::{orientation_histogram::generate_keypoints_from_extrema,local_image_descriptor::{is_rotated_keypoint_within_image,LocalImageDescriptor},feature_vector::FeatureVector,keypoint::KeyPoint};
use vision::visualize::{draw_square,visualize_pyramid_feature_with_orientation};

fn main() {
    let image_name = "lenna_90";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let converted_file_out_path = format!("{}{}_squares.{}",image_out_folder,image_name,image_format);
    let converted_refined_file_out_path = format!("{}{}_refined_squares.{}",image_out_folder,image_name,image_format);
    let orientation_file_out_path = format!("{}{}_orientation.{}",image_out_folder,image_name,image_format);


    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
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
    let octave_level = 1;
    let sigma_level = 1;
    let octave = &pyramid.octaves[octave_level];


    let mut display = Image::from_matrix(&pyramid.octaves[octave_level].images[0].buffer,pyramid.octaves[octave_level].images[0].original_encoding,false);
    let mut orientation_display =  Image::from_matrix(&pyramid.octaves[0].images[0].buffer,pyramid.octaves[octave_level].images[0].original_encoding,false);
    let mut refined_display = Image::from_matrix(&pyramid.octaves[octave_level].images[0].buffer,pyramid.octaves[octave_level].images[0].original_encoding,false);

    let x_step = 1;
    let y_step = 1;



    let features = sift_feature::detect_sift_feature(octave,sigma_level,x_step, y_step);
    let refined_features = sift_feature::sift_feature_refinement(&features, octave, &runtime_params);
    let keypoints = refined_features.iter().map(|x| generate_keypoints_from_extrema(octave,octave_level, x, &runtime_params)).flatten().collect::<Vec<KeyPoint>>();
    let descriptors = keypoints.iter().filter(|x| is_rotated_keypoint_within_image(octave, x)).map(|x| LocalImageDescriptor::new(octave,x)).collect::<Vec<LocalImageDescriptor>>();
    let feature_vectors = descriptors.iter().map(|x| FeatureVector::new(x,octave_level)).collect::<Vec<FeatureVector>>();


    let number_of_features = feature_vectors.len();
    let number_of_refined_features = refined_features.len();
    let number_of_keypoints = keypoints.len();

    let rows = octave.images[0].buffer.nrows();
    let cols = octave.images[0].buffer.ncols();
    let size = rows*cols;
    let percentage = number_of_features as f32/size as f32;
    let refined_percentage = number_of_refined_features as f32/size as f32;
    let keypoint_percentage = number_of_keypoints as f32/size as f32;

    println!("Features: {} out of {}, ({}%)",number_of_features, size, percentage);
    println!("Refined Features: {} out of {}, ({}%)",number_of_refined_features, size, refined_percentage);
    println!("Keypoints: {} out of {}, ({}%)",number_of_keypoints, size, keypoint_percentage);


    for keypoint in keypoints {
        visualize_pyramid_feature_with_orientation(&mut orientation_display, &keypoint,keypoint.octave_level, 64.0);
    }

    for feature in features {
        let x = feature.x.trunc() as usize;
        let y = feature.y.trunc() as usize;

        assert!(x < display.buffer.ncols());
        assert!(y < display.buffer.nrows());

        draw_square(&mut display, x, y, 1);
    }

    for feature in refined_features {
        let x = feature.x.trunc() as usize;
        let y = feature.y.trunc() as usize;

        assert!(x < display.buffer.ncols());
        assert!(y < display.buffer.nrows());

        draw_square(&mut refined_display, x, y, 1);
    }

    let new_image = display.to_image();
    let refined_new_image = refined_display.to_image();
    let orientation_new_image = orientation_display.to_image();

    new_image.save(converted_file_out_path).unwrap();
    refined_new_image.save(converted_refined_file_out_path).unwrap();
    orientation_new_image.save(orientation_file_out_path).unwrap();





}