extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::io::tum_loader;
use vision::pyramid::rgbd::{build_rgbd_pyramid,rgbd_runtime_parameters::RGBDRuntimeParameters};
use vision::vo::{dense_direct,dense_direct::{dense_direct_runtime_parameters::DenseDirectRuntimeParameters}};

fn main() {
    //let root_path = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole";
    let root_path = "C:/Users/Marc/Workspace/Datasets/ETH/vfr_pinhole";
    
    let image_idx = 99;
    let image_2_idx = 100;
    let intrinsics = "intrinsics";
    let ts_names = "images";
    let ground_truths = "groundtruth";

    let depth_image_format = "depth";
    let color_image_format = "png";
    let text_format = "txt";

    let out_folder = "output/";
    let depth_image_folder = format!("{}/{}",root_path,"data/depth/");
    let color_image_folder = format!("{}/{}",root_path,"data/img/");
    let info_folder = format!("{}/{}",root_path,"info/");
    let ts_name_path = format!("{}{}.{}",info_folder,ts_names, text_format);
    let ground_truths_path = format!("{}{}.{}",info_folder,ground_truths, text_format);

    let ts_names = tum_loader::load_timestamps_and_names(&Path::new(&ts_name_path));
    let ground_truths = tum_loader::load_ground_truths(&Path::new(&ground_truths_path));

    let image = &ts_names[image_idx].1;
    let ground_truth_1 = ground_truths[image_idx];
    let image_2 = &ts_names[image_2_idx].1;
    let ground_truth_2 = ground_truths[image_2_idx];


    let color_image_path = format!("{}{}.{}",color_image_folder,image, color_image_format);
    let depth_image_path = format!("{}{}.{}",depth_image_folder,image, depth_image_format);

    let color_2_image_path = format!("{}{}.{}",color_image_folder,image_2, color_image_format);
    let depth_2_image_path = format!("{}{}.{}",depth_image_folder,image_2, depth_image_format);

    let intrinsics_path = format!("{}{}.{}",info_folder,intrinsics, text_format);



    let negate_values = true;
    let invert_focal_lengths = true;
    let invert_y = true;
    let invert_grad_x = true;
    let invert_grad_y = true;
    let debug = true;

    //TODO: euclidean depth
    let depth_display = tum_loader::load_depth_image(&Path::new(&depth_image_path),negate_values, true);
    let gray_display = tum_loader::load_image_as_gray(&Path::new(&color_image_path), false, invert_y);

    let depth_2_display = tum_loader::load_depth_image(&Path::new(&depth_2_image_path), negate_values, true);
    let gray_2_display = tum_loader::load_image_as_gray(&Path::new(&color_2_image_path), false, invert_y);

    let pinhole_camera = tum_loader::load_intrinsics_as_pinhole(&Path::new(&intrinsics_path), invert_focal_lengths);

    println!("{:?}",pinhole_camera.projection);
    println!("{},{}",image,image_2);

    let pyramid_parameters = RGBDRuntimeParameters{
        sigma: 0.01,
        use_blur: false,
        blur_radius: 3.0,
        octave_count: 1,
        min_image_dimensions: (50,50),
        invert_grad_x,
        invert_grad_y
    };

    let image_pyramid_1 = build_rgbd_pyramid(gray_display,depth_display,&pyramid_parameters);
    let image_pyramid_2 = build_rgbd_pyramid(gray_2_display,depth_2_display,&pyramid_parameters);

    let vo_parameters = DenseDirectRuntimeParameters{
        max_iterations: 200,
        eps: 1e-8,
        step_size: 0.1,
        max_norm_eps: 1e-12,
        delta_eps: 1e-9,
        tau: 1e-3,
        debug,
        lm: false
    };

    dense_direct::run(&image_pyramid_1, &image_pyramid_2,&pinhole_camera , &vo_parameters);

    println!("Groundtruth Pos {}: T: {:?}, Q: {:?}",image_idx, ground_truth_1.0,ground_truth_1.1);
    println!("Groundtruth Pos {}: T: {:?}, Q: {:?}",image_2_idx, ground_truth_2.0,ground_truth_2.1);




}