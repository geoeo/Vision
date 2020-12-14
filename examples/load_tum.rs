extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::io::tum_loader;

fn main() {
    let image_name = "1305031453.374112";
    let image_2_name = "img0002_0";
    let intrinsics_name = "intrinsics";
    let depth_image_format = "png";
    let color_image_format = "png";
    let intrinsics_format = "txt";
    let depth_image_folder = "C:/Users/Marc/Workspace/Datasets/TUM/rgbd_dataset_freiburg1_desk/depth/";
    let color_image_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_data/data/img/";
    let intrinsics_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_info/info/";
    let image_out_folder = "output/";

    let depth_image_path = format!("{}{}.{}",depth_image_folder,image_name, depth_image_format);


    let depth_display = tum_loader::load_depth_image(&Path::new(&depth_image_path), false, false);

    let converted_file_out_path = format!("{}{}_out.png",image_out_folder,image_name);
    let new_image = depth_display.to_image();
    new_image.save(converted_file_out_path).unwrap();






}