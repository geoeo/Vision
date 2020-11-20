extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::io::tum_loader;

fn main() {

    let image_name = "img0008_0";
    let image_format = "depth";
    //let image_folder = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole/rpg_urban_pinhole_depthmaps/data/depth/";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let depth_image_path = format!("{}{}.{}",image_folder,image_name, image_format);
    let converted_file_out_path = format!("{}{}_out.png",image_out_folder,image_name);


    let display = tum_loader::load_depth_image(&Path::new(&depth_image_path));

    let new_image = display.to_image();

    new_image.save(converted_file_out_path).unwrap();


}