extern crate image as image_rs;
extern crate vision;

use std::path::Path;
use vision::io::{load_depth_image_from_csv,load_image_as_gray,d455_loader};

fn main() {
    let color_image_name = "_Color_1609709584112.90991210937500";
    let depth_image_name = "_Depth_1609709584079.51391601562500";
    //let intrinsics_name = "intrinsics";
    let depth_image_format = "csv";
    let color_image_format = "png";
    let intrinsics_format = "txt";
    let depth_image_folder = "C:/Users/Marc/Workspace/Datasets/D455/simple_trans/depth/";
    let color_image_folder = "C:/Users/Marc/Workspace/Datasets/D455/simple_trans/rgb/";
    //let intrinsics_folder = "";
    let image_out_folder = "output/";

    let color_image_path = format!("{}{}.{}",color_image_folder,color_image_name, color_image_format);
    let depth_image_path = format!("{}{}.{}",depth_image_folder,depth_image_name, depth_image_format);

    //let color_2_image_path = format!("{}{}.{}",color_image_folder,image_2_name, color_image_format);
    //let depth_2_image_path = format!("{}{}.{}",depth_image_folder,image_2_name, depth_image_format);

    //let intrinsics_path = format!("{}{}.{}",intrinsics_folder,intrinsics_name, intrinsics_format);

    let gray_display = load_image_as_gray(&Path::new(&color_image_path), false, false);
    let depth_display = load_depth_image_from_csv(&Path::new(&depth_image_path), false, false, 640,480, 1.0, false, true);


    //let depth_2_display = load_depth_image(&Path::new(&depth_2_image_path), false,false, 5000.0);
    //let gray_2_display = load_image_as_gray(&Path::new(&color_2_image_path), false, true);

    //let pinhole_camera = eth_loader::load_intrinsics_as_pinhole(&Path::new(&intrinsics_path), false);



    let converted_file_out_path = format!("{}{}_out.png",image_out_folder,depth_image_name);
    let new_image = depth_display.to_image();
    new_image.save(converted_file_out_path).unwrap();


}