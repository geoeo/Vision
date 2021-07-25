extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::pyramid::orb::{
    build_orb_pyramid,generate_feature_pyramid,generate_feature_descriptor_pyramid,  orb_runtime_parameters::OrbRuntimeParameters, generate_matches,generate_matches_between_pyramid

};
use vision::visualize::{visualize_pyramid_feature_with_orientation, display_matches_for_pyramid};
use vision::matching::brief_descriptor::BriefDescriptor;
use vision::image::Image;

fn main() {

    //let image_name = "lenna";
    //let image_name_2 = "lenna_90";

    //let image_name = "beaver";
    //let image_name_2 = "beaver_90";

    //let image_name = "beaver";
    //let image_name_2 = "beaver_scaled_50";

    // let image_name = "cereal_1_scaled_25";
    // let image_name_2 = "cereal_2_scaled_25";

    //let image_name = "cereal_1_scaled_25";
    //let image_name_2 = "cereal_2_far_scaled_25";

    //let image_name = "cereal_1_scaled_25";
    //let image_name_2 = "cereal_tilted_scaled_25";

    //let image_name = "cereal_1_scaled_25";
    //let image_name_2 = "cereal_occluded_scaled_25";

    //let image_name = "cereal_1_scaled_25";
    //let image_name_2 = "cereal_far_scaled_25";

    let image_name = "ba_slow_3";
    let image_name_2 = "ba_slow_4";

    //let image_name = "ba_slow_4";
    let image_name_3 = "ba_slow_5";




    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let image_path_3 = format!("{}{}.{}",image_folder,image_name_3, image_format);


    println!("{}, {}",image_path,image_path_2);

    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();
    let gray_image_3 = image_rs::open(&Path::new(&image_path_3)).unwrap().to_luma8();

    let image = Image::from_gray_image(&gray_image, false, false, Some(image_name.to_string()));
    let image_2 = Image::from_gray_image(&gray_image_2, false, false, Some(image_name_2.to_string()));
    let image_3 = Image::from_gray_image(&gray_image_3, false, false, Some(image_name_3.to_string()));



    //TODO: recheck maximal suppression, take best corers for all windows across all pyramid levels
    // https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj2/html/agartia3/index.html
    let pyramid_scale = 1.2; // opencv default is 1.2
    let runtime_params = OrbRuntimeParameters {
        pyramid_scale: pyramid_scale,
        min_image_dimensions: (20,20),
        sigma: 2.0,
        blur_radius: 3.0,
        max_features_per_octave: 5,
        max_features_per_octave_scale: 1.2,
        octave_count: 8, // opencv default is 8
        harris_k: 0.04,
        harris_window_size: 5, 
        fast_circle_radius: 3,
        fast_threshold_factor: 0.2,
        fast_consecutive_pixels: 12,
        fast_features_per_grid: 3,
        fast_grid_size: (15,15),  
        fast_grid_size_scale_base: 1.2,
        fast_offsets: (12,12),
        fast_offset_scale_base: 1.2,
        brief_features_to_descriptors: 128,
        brief_n: 256,
        brief_s: 31,
        brief_s_scale_base: pyramid_scale,
        brief_matching_min_threshold: 256/2, 
        brief_lookup_table_step: 30.0,
        brief_sampling_pattern_seed: 0x0DDB1A5ECBAD5EEDu64,
        brief_use_opencv_sampling_pattern: true
    };



    let matches = generate_matches(vec!((&image, &runtime_params, &image_2, &runtime_params), ((&image_2, &runtime_params, &image_3, &runtime_params))));



    println!("matching complete");


    //TODO: make this work with images of different sizes
    println!("{}",matches.len());


    let display = Image::from_gray_image(&gray_image, false, false, None); 
    let display_2 = Image::from_gray_image(&gray_image_2, false, false, None); 
    let display_3 = Image::from_gray_image(&gray_image_3, false, false, None); 


    let match_display_1_2 = display_matches_for_pyramid(&display, &display_2, &matches[0], true, display.buffer.max()/2.0, runtime_params.pyramid_scale);
    let match_display_2_3 = display_matches_for_pyramid(&display_2, &display_3, &matches[1], true, display.buffer.max()/2.0, runtime_params.pyramid_scale);


    match_display_1_2.to_image().save(format!("{}{}_orb_ba.{}",image_out_folder,image_name,image_format)).unwrap();
    match_display_2_3.to_image().save(format!("{}{}_orb_ba.{}",image_out_folder,image_name_2,image_format)).unwrap();


}