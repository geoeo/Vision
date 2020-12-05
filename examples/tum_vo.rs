extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::io::tum_loader;
use vision::pyramid::rgbd::{build_rgbd_pyramid,rgbd_runtime_parameters::RGBDRuntimeParameters};
use vision::vo::{dense_direct,dense_direct::{dense_direct_runtime_parameters::DenseDirectRuntimeParameters}};
use vision::camera::pinhole::Pinhole;

fn main() {
    let image_name = "img0001_0";
    let image_2_name = "img0002_0";
    let intrinsics_name = "intrinsics";
    let depth_image_format = "depth";
    let color_image_format = "png";
    let intrinsics_format = "txt";
    let depth_image_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_depthmaps/data/depth/";
    let color_image_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_data/data/img/";
    let intrinsics_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_info/info/";
    let image_out_folder = "output/";

    let color_image_path = format!("{}{}.{}",color_image_folder,image_name, color_image_format);
    let depth_image_path = format!("{}{}.{}",depth_image_folder,image_name, depth_image_format);

    let color_2_image_path = format!("{}{}.{}",color_image_folder,image_2_name, color_image_format);
    let depth_2_image_path = format!("{}{}.{}",depth_image_folder,image_2_name, depth_image_format);

    let intrinsics_path = format!("{}{}.{}",intrinsics_folder,intrinsics_name, intrinsics_format);


    let depth_display = tum_loader::load_depth_image(&Path::new(&depth_image_path));
    let gray_display = tum_loader::load_image_as_gray(&Path::new(&color_image_path));

    let depth_2_display = tum_loader::load_depth_image(&Path::new(&depth_2_image_path));
    let gray_2_display = tum_loader::load_image_as_gray(&Path::new(&color_2_image_path));

    let pinhole_camera = tum_loader::load_intrinsics_as_pinhole(&Path::new(&intrinsics_path));
    //let pinhole_camera = Pinhole::new(1.0, 1.0, 0.0, 0.0);

    println!("{:?}",pinhole_camera.projection);

    let pyramid_parameters = RGBDRuntimeParameters{
        sigma: 0.1,
        use_blur: false,
        blur_radius: 3.0,
        octave_count: 1,
        min_image_dimensions: (50,50)
    };

    let image_pyramid_1 = build_rgbd_pyramid(gray_display,depth_display,&pyramid_parameters);
    let image_pyramid_2 = build_rgbd_pyramid(gray_2_display,depth_2_display,&pyramid_parameters);

    let vo_parameters = DenseDirectRuntimeParameters{
        max_iterations: 200,
        eps: 1e-8
    };

    dense_direct::run(&image_pyramid_1, &image_pyramid_2,&pinhole_camera , &vo_parameters);





}