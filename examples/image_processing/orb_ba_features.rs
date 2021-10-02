extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;

use std::fs;
use color_eyre::eyre::Result;
use std::path::Path;
use vision::image::pyramid::orb::{orb_runtime_parameters::OrbRuntimeParameters, generate_matches};
use vision::image::features::{Match,Feature,orb_feature::OrbFeature};
use vision::image::Image;
use vision::sensors::camera::pinhole::Pinhole;


fn main() ->Result<()> {

    color_eyre::install()?;

    let image_name_1 = "ba_slow_1";
    let image_name_2 = "ba_slow_2";

    let image_name_3 = "ba_slow_3";
    let image_name_4 = "ba_slow_4";



    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path_1 = format!("{}{}.{}",image_folder,image_name_1, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let image_path_3 = format!("{}{}.{}",image_folder,image_name_3, image_format);
    let image_path_4 = format!("{}{}.{}",image_folder,image_name_4, image_format);

    let gray_image_1 = image_rs::open(&Path::new(&image_path_1)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();
    let gray_image_3 = image_rs::open(&Path::new(&image_path_3)).unwrap().to_luma8();
    let gray_image_4 = image_rs::open(&Path::new(&image_path_4)).unwrap().to_luma8();

    let image_1 = Image::from_gray_image(&gray_image_1, false, false, Some(image_name_1.to_string()));
    let image_2 = Image::from_gray_image(&gray_image_2, false, false, Some(image_name_2.to_string()));
    let image_3 = Image::from_gray_image(&gray_image_3, false, false, Some(image_name_3.to_string()));
    let image_4 = Image::from_gray_image(&gray_image_4, false, false, Some(image_name_4.to_string()));



    //TODO: recheck maximal suppression, take best corers for all windows across all pyramid levels
    // https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj2/html/agartia3/index.html
    let pyramid_scale = 1.2; // opencv default is 1.2
    let runtime_params = OrbRuntimeParameters {
        pyramid_scale: pyramid_scale,
        min_image_dimensions: (20,20),
        sigma: 2.0,
        blur_radius: 3.0,
        max_features_per_octave: 15, // 15
        max_features_per_octave_scale: 1.2,
        octave_count: 3, // opencv default is 8
        harris_k: 0.04,
        harris_window_size: 5,  // 5
        fast_circle_radius: 3, //3 
        fast_threshold_factor: 0.2,
        fast_consecutive_pixels: 12, // 12
        fast_features_per_grid: 3, // 3
        fast_grid_size: (15,15),  // 15
        fast_grid_size_scale_base: 1.2,
        fast_offsets: (12,12),
        fast_offset_scale_base: 1.2,
        brief_features_to_descriptors: 128, // 128
        brief_n: 256,
        brief_s: 31,
        brief_s_scale_base: pyramid_scale,
        brief_matching_min_threshold: 256/2, //256/2
        brief_lookup_table_step: 30.0,
        brief_sampling_pattern_seed: 0x0DDB1A5ECBAD5EEDu64,
        brief_use_opencv_sampling_pattern: true
    };

    //TODO: also serialize cameras (ordering + intrinsics) into the yaml file

    //TODO: camera intrinsics 
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, false);
    let intensity_camera_2 = intensity_camera_1.clone();
    let intensity_camera_3 = intensity_camera_1.clone();
    let intensity_camera_4 = intensity_camera_1.clone();

    let cameras = vec!(intensity_camera_1,intensity_camera_2,intensity_camera_3,intensity_camera_4);
    //let cameras = vec!(intensity_camera_1,intensity_camera_2);

    let image_pairs = vec!((&image_1, &runtime_params, &image_2, &runtime_params), ((&image_3, &runtime_params, &image_4, &runtime_params)));
    //let image_pairs = vec!((&image_1, &runtime_params, &image_2, &runtime_params));
    println!("start matching...");
    let matches = generate_matches(&image_pairs); //TODO: save matches to file for loading
    println!("matching complete");
    

    let s = serde_yaml::to_string(&matches)?;


    fs::write("D:/Workspace/Rust/Vision/output/orb_ba_matches.txt", s).expect("Unable to write file");

    // let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches.txt").expect("Unable to read file");
    // let deserialized_matches: Vec<Vec<Match<OrbFeature>>> = serde_yaml::from_str(&orb_matches_read)?;


    // for a in deserialized_matches{
    //     for b in a {
    //         let (idx,feat) = b.feature_one;
    //         println!("{}",feat.get_x_image());
    //     }
    // }


    Ok(())

}