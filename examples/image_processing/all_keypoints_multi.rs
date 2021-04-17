extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::pyramid::sift::{build_sift_pyramid, sift_runtime_params::SiftRuntimeParams,keypoints_from_pyramid};
use vision::image::Image;
use vision::visualize::visualize_pyramid_feature_with_orientation;

fn main() {
    


    let image_names = vec!["circles","blur","blur_rotated"];


    for image_name in image_names {

        let image_format = "png";
        let image_folder = "images/";
        let image_out_folder = "output/";
        let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    
        let converted_file_out_path = format!("{}{}_keypoints_all.{}",image_out_folder,image_name,image_format);
    
        println!("Processing Image: {}", image_name);
    
        let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma8();
        let image = Image::from_gray_image(&gray_image, false, false);
        let mut display = Image::from_gray_image(&gray_image, false, false);
        
        
        //TODO: move inital blur params here
        let runtime_params = SiftRuntimeParams {
            min_image_dimensions: (25,25),
            blur_half_factor: 3.0, //TODO: lowering <= 4 this causes algorithm to become unstable
            orientation_histogram_window_factor: 1.0, //TODO: investigate
            edge_r: 10.0,
            contrast_r: 0.03,
            sigma_initial: 1.6,
            sigma_in: 0.5,
            octave_count: 6,
            sigma_count: 3
        };

        //TODO: experiment with blur half width and pyramid params
        let pyramid = build_sift_pyramid(image,&runtime_params);
    
        let all_keypoints = keypoints_from_pyramid(&pyramid, &runtime_params);
        //let all_keypoints = keypoints_from_octave(&pyramid,2, &runtime_params);
    
        let number_of_features = all_keypoints.len();
    
        let rows = pyramid.octaves[0].images[0].buffer.nrows();
        let cols = pyramid.octaves[0].images[0].buffer.ncols();
        let size = rows*cols;
        let percentage = number_of_features as f32/size as f32;
    
        println!("Keypoints from Image: {} out of {}, ({}%)",number_of_features, size, percentage);
    
        for keypoint in all_keypoints {
            visualize_pyramid_feature_with_orientation(&mut display, &keypoint, keypoint.octave_level, 64.0);
        }
    
    
        let new_image = display.to_image();
    
        new_image.save(converted_file_out_path).unwrap();

    }







}