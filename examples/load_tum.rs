extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::io::tum_loader;

fn main() {
    let depth_image_name = "img0001_0";
    let intrinsics_name = "intrinsics";
    let depth_image_format = "depth";
    let intrinsics_format = "txt";
    let depth_image_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_depthmaps/data/depth/";
    let intrinsics_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_info/info/";
    let image_out_folder = "output/";

    let depth_image_path = format!("{}{}.{}",depth_image_folder,depth_image_name, depth_image_format);
    let depth_converted_file_out_path = format!("{}{}_out.png",image_out_folder,depth_image_name);

    let intrinsics_path = format!("{}{}.{}",intrinsics_folder,intrinsics_name, intrinsics_format);
    println!("{}",intrinsics_path);

    let depth_display = tum_loader::load_depth_image(&Path::new(&depth_image_path));
    let intrinsics = tum_loader::load_intrinsics(&Path::new(&intrinsics_path));


    let new_depth_image = depth_display.to_image();
    new_depth_image.save(depth_converted_file_out_path).unwrap();


}