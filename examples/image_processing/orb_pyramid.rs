extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::pyramid::orb::{build_orb_pyramid,generate_feature_pyramid, generate_feature_descriptor_pyramid,  orb_runtime_parameters::OrbRuntimeParameters};
use vision::visualize::visualize_pyramid_feature_with_orientation;
use vision::matching::brief_descriptor::BriefDescriptor;
use vision::image::Image;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let converted_file_out_path = format!("{}{}_features_with_descriptors.{}",image_out_folder,image_name,image_format);

    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma8();
    let image = Image::from_gray_image(&gray_image, false, false);

    let mut display = Image::from_gray_image(&gray_image, false, false); 
    let display_max = display.buffer.max();

    let runtime_params = OrbRuntimeParameters {
        min_image_dimensions: (50,50),
        sigma: 0.5,
        blur_radius: 5.0,
        max_features_per_octave: std::usize::MAX,
        max_features_per_octave_scale: 1.25,
        octave_count: 3,
        harris_k: 0.04,
        fast_circle_radius: 3,
        fast_threshold_factor: 0.2,
        fast_consecutive_pixels: 12,
        fast_grid_size: (10,10),
        fast_grid_size_scale_base: 2.0,
        fast_offsets: (20,20),
        fast_offset_scale_base: 1.25,
        brief_features_to_descriptors: 128,
        brief_n: 256,
        brief_s: 31,
        brief_s_scale_base: 2.0,
        brief_matching_min_threshold: 256/4,
        brief_lookup_table_step: 8.0,
        brief_sampling_pattern_seed: 0x0DDB1A5ECBAD5EEDu64,
        brief_use_opencv_sampling_pattern: true
    };
    
    //let sample_lookup_table = BriefDescriptor::generate_sample_lookup_tables(runtime_params.brief_n, runtime_params.brief_s);
    let sample_lookup_pyramid = BriefDescriptor::generate_sample_lookup_table_pyramid(&runtime_params,runtime_params.octave_count);

    let pyramid = build_orb_pyramid(image, &runtime_params);
    let feature_pyramid = generate_feature_pyramid(&pyramid, &runtime_params);
    let feautre_descriptors = generate_feature_descriptor_pyramid(&pyramid,&feature_pyramid,&sample_lookup_pyramid,&runtime_params);

    // for i in 0..pyramid.octaves.len() {
    //     let octave = &pyramid.octaves[i];
    //     let image = &octave.images[0];
    //     let gray_image  = image.to_image();

    //     let name = format!("orb_image_{}",i);
    //     let file_path = format!("{}{}.{}",image_out_folder,name,image_format);
    //     gray_image.save(file_path).unwrap();
    // }

    for octave_index in 0..feautre_descriptors.octaves.len() {
        let octave = &feautre_descriptors.octaves[octave_index];
        for (feature,_) in octave {
            visualize_pyramid_feature_with_orientation(&mut display, feature, octave_index, display_max/2.0);
        }
    }

    let new_image = display.to_image();

    new_image.save(converted_file_out_path).unwrap();

}