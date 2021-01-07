extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;

use na::{Vector3,UnitQuaternion};
use vision::io::{tum_loader,loading_parameters::LoadingParameters};
use vision::Float;

fn main() {
    let image_name = "tum_out";
    let image_out_folder = "output";

    // let depth_image_path = format!("{}{}.{}",depth_image_folder,image_name, depth_image_format);
    // let ground_truth_path = format!("{}{}.{}",info_folder,ground_truth_name, info_format);


    // let depth_display = tum_loader::load_depth_image(&Path::new(&depth_image_path), false, false);
    // let (timestamps,ground_truths) = tum_loader::load_timestamps_ground_truths(&Path::new(&ground_truth_path));



    let root_path = "C:/Users/Marc/Workspace/Datasets/TUM/rgbd_dataset_freiburg1_desk";




    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :1,
        negate_depth_values :true,
        invert_focal_lengths :true,
        invert_y :false,
        gt_alignment_rot: UnitQuaternion::<Float>::identity()
    };

    let dataset = tum_loader::Dataset::FR1;

    let loaded_data = tum_loader::load(root_path, &loading_parameters, &dataset);


    let converted_file_out_path = format!("{}/{}_out.png",image_out_folder,image_name);
    let new_image = loaded_data.source_gray_images[0].to_image();
    new_image.save(converted_file_out_path).unwrap();







}