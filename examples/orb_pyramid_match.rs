extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::pyramid::orb::{build_orb_pyramid,generate_feature_pyramid,generate_feature_descriptor_pyramid,  orb_runtime_parameters::OrbRuntimeParameters, generate_match_pyramid};
use vision::visualize::{visualize_pyramid_feature_with_orientation,display_matches_for_octave, display_matches_for_pyramid};
use vision::matching::brief_descriptor::BriefDescriptor;
use vision::features::Oriented;
use vision::image::Image;
use vision::Float;

fn main() {

    let image_name = "lenna";
    let image_name_2 = "lenna_90";
    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let converted_file_out_path = format!("{}{}_orb_matches_all.{}",image_out_folder,image_name,image_format);

    let gray_image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();

    let image = Image::from_gray_image(&gray_image, false, false);
    let image_2 = Image::from_gray_image(&gray_image_2, false, false);

    let display = Image::from_gray_image(&gray_image, false, false); 
    let display_2 = Image::from_gray_image(&gray_image_2, false, false); 

    let runtime_params = OrbRuntimeParameters {
        min_image_dimensions: (50,50),
        sigma: 2.0,
        blur_radius: 4.0,
        max_features_per_octave: 100,
        octave_count: 3,
        harris_k: 0.04,
        fast_circle_radius: 3,
        fast_threshold_factor: 0.2,
        fast_consecutive_pixels: 12,
        fast_grid_size: (10,10),
        fast_offsets: (0,0),
        brief_n: 256,
        brief_s: 31,
        brief_matching_min_threshold: 256/2
    };
    
    let sample_lookup_table = BriefDescriptor::generate_sample_lookup_tables(runtime_params.brief_n, runtime_params.brief_s);

    let pyramid = build_orb_pyramid(image, &runtime_params);
    let feature_pyramid = generate_feature_pyramid(&pyramid, &runtime_params);
    let feature_descriptor_pyramid_a = generate_feature_descriptor_pyramid(&pyramid,&feature_pyramid,&sample_lookup_table,&runtime_params);

    let pyramid_2 = build_orb_pyramid(image_2, &runtime_params);
    let feature_pyramid_2 = generate_feature_pyramid(&pyramid_2, &runtime_params);
    let feature_descriptor_pyramid_b = generate_feature_descriptor_pyramid(&pyramid_2,&feature_pyramid_2,&sample_lookup_table,&runtime_params);

    let match_pyramid = generate_match_pyramid(&feature_descriptor_pyramid_a,&feature_descriptor_pyramid_b, &runtime_params);

    for i in 0..pyramid.octaves.len() {
        let avg_orientation_a = &feature_pyramid.octaves[i].iter().fold(0.0,|acc,x| acc + x.get_orientation())/feature_pyramid.octaves[i].len() as Float;
        let avg_orientation_b = &feature_pyramid_2.octaves[i].iter().fold(0.0,|acc,x| acc + x.get_orientation())/feature_pyramid_2.octaves[i].len() as Float;

        println!("octave: {}, avg orientation for a: {}, avg orientation for b: {}",i,avg_orientation_a,avg_orientation_b);
        println!("octave: {}, difference in orientation: {}",i,(avg_orientation_a-avg_orientation_b).abs());

    }

    for i in 0..pyramid.octaves.len() {
        let display_a = &pyramid.octaves[i].images[0];
        let display_b = &pyramid_2.octaves[i].images[0];

        let matches = &match_pyramid.octaves[i];
        let radius = (pyramid.octaves.len()-i) as Float *10.0; 
        let match_dispay = display_matches_for_octave(display_a, display_b, matches,radius, true, display_a.buffer.max()/2.0); //TODO: fix this. values are getting really large
        let gray_image  = match_dispay.to_image();

        let name = format!("orb_match_{}",i);
        let file_path = format!("{}{}.{}",image_out_folder,name,image_format);
        gray_image.save(file_path).unwrap();
    }


    let pyramid_match_display = display_matches_for_pyramid(&display, &display_2, &match_pyramid.octaves, true, display.buffer.max()/2.0);



    let new_image = pyramid_match_display.to_image();
    new_image.save(converted_file_out_path).unwrap();

}