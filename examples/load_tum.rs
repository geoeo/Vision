extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::io::tum_loader;

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


    let depth_display = tum_loader::load_depth_image(&Path::new(&depth_image_path), false);
    let gray_display = tum_loader::load_image_as_gray(&Path::new(&color_image_path), false, true);

    let depth_2_display = tum_loader::load_depth_image(&Path::new(&depth_2_image_path), false);
    let gray_2_display = tum_loader::load_image_as_gray(&Path::new(&color_2_image_path), false, true);

    let pinhole_camera = tum_loader::load_intrinsics_as_pinhole(&Path::new(&intrinsics_path));



    let converted_file_out_path = format!("{}{}_out.png",image_out_folder,image_name);
    let new_image = gray_2_display.to_image();
    new_image.save(converted_file_out_path).unwrap();


}