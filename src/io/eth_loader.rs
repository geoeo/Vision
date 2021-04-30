extern crate nalgebra as na;
extern crate image as image_rs;

use na::{RowDVector,DMatrix, Vector3, Quaternion, UnitQuaternion};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader,Read,BufRead};

use crate::Float;
use crate::io::{image_loading_parameters::ImageLoadingParameters, parse_to_float};
use crate::image::Image;
use crate::sensors::camera::{camera_data_frame::CameraDataFrame,pinhole::Pinhole};
use crate::io::{load_image_as_gray, load_depth_image_from_csv};



pub fn load(root_path: &str, parameters: &ImageLoadingParameters) -> CameraDataFrame {
    let intrinsics = "intrinsics";
    let ts_names = "images";
    let ground_truths = "groundtruth";

    let depth_image_format = "depth";
    let color_image_format = "png";
    let text_format = "txt";

    let depth_image_folder = format!("{}/{}",root_path,"data/depth/");
    let color_image_folder = format!("{}/{}",root_path,"data/img/");
    let info_folder = format!("{}/{}",root_path,"info/");
    let ts_name_path = format!("{}{}.{}",info_folder,ts_names, text_format);
    let ground_truths_path = format!("{}{}.{}",info_folder,ground_truths, text_format);

    let ts_names = load_timestamps_and_names(&Path::new(&ts_name_path));
    let ground_truths = load_ground_truths(&Path::new(&ground_truths_path), &parameters.gt_alignment_rot);
    let intrinsics_path = format!("{}{}.{}",info_folder,intrinsics, text_format);


    let source_indices = (parameters.starting_index..parameters.starting_index+parameters.count).step_by(parameters.step);
    let target_indices = source_indices.clone().map(|x| x + parameters.step); //TODO: check out of range



    assert_eq!(ground_truths.len(),ts_names.len());

    let intensity_camera = load_intrinsics_as_pinhole(&Path::new(&intrinsics_path), parameters.invert_focal_lengths);
    let depth_camera = load_intrinsics_as_pinhole(&Path::new(&intrinsics_path), parameters.invert_focal_lengths);

    CameraDataFrame {
        source_timestamps: source_indices.clone().map(|s| ts_names[s].0).collect::<Vec<Float>>(),
        target_timestamps: target_indices.clone().map(|t| ts_names[t].0).collect::<Vec<Float>>(),
        source_gray_images: source_indices.clone().map(|s| {
            let source = &ts_names[s].1;
            let color_source_image_path = format!("{}{}.{}",color_image_folder,source, color_image_format);
            load_image_as_gray(&Path::new(&color_source_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        source_depth_images: source_indices.clone().map(|s| {
            let source = &ts_names[s].1;
            let depth_source_image_path = format!("{}{}.{}",depth_image_folder,source, depth_image_format);
            load_depth_image_from_csv(&Path::new(&depth_source_image_path),parameters.negate_depth_values, true, parameters.image_width,parameters.image_height, 1.0, false,parameters.set_default_depth, &None)
        }).collect::<Vec<Image>>(),
        target_gray_images: target_indices.clone().map(|t| {
            let target = &ts_names[t].1;
            let color_target_image_path = format!("{}{}.{}",color_image_folder,target, color_image_format);
            load_image_as_gray(&Path::new(&color_target_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        target_depth_images:  target_indices.clone().map(|t| {
            let target = &ts_names[t].1;
            let depth_target_image_path = format!("{}{}.{}",depth_image_folder,target, depth_image_format);
            load_depth_image_from_csv(&Path::new(&depth_target_image_path), parameters.negate_depth_values, true, parameters.image_width,parameters.image_height, 1.0, false,parameters.set_default_depth, &None)
        }).collect::<Vec<Image>>(),
        intensity_camera,
        depth_camera,
        target_gt_poses: Some(target_indices.clone().map(|t| {
            ground_truths[t]
        }).collect::<Vec<(Vector3<Float>,Quaternion<Float>)>>()),
        source_gt_poses: Some(source_indices.clone().map(|s| {
            ground_truths[s]
        }).collect::<Vec<(Vector3<Float>,Quaternion<Float>)>>())
    }
}

fn load_timestamps_and_names(file_path: &Path)-> Vec<(Float,String)> {
    let file = File::open(file_path).expect("load_timestamps_and_names failed");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    let mut timestamps_and_names = Vec::<(Float,String)>::new();

    for line in lines {
        let contents = line.unwrap();
        let values = contents.trim().split(" ").collect::<Vec<&str>>();
        let name = String::from(values[2].split(|c| c == '/' || c == '.').collect::<Vec<&str>>()[1]);
        let ts = parse_to_float(values[1], false);
        timestamps_and_names.push((ts,name));
    }

    timestamps_and_names

}

pub fn load_ground_truths(file_path: &Path,  gt_alignment_rot: &UnitQuaternion<Float>) -> Vec<(Vector3<Float>,Quaternion<Float>)> {
    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    let mut ground_truths = Vec::<(Vector3<Float>,Quaternion<Float>)>::new();

    for line in lines {
        let contents = line.unwrap();
        let values = contents.trim().split(" ").map(|x| parse_to_float(x,false)).collect::<Vec<Float>>();
        let t = gt_alignment_rot*Vector3::new(values[1],values[2],values[3]);
        let quat = gt_alignment_rot.quaternion()*Quaternion::<Float>::new(values[7],values[4],values[5], values[6]);
        ground_truths.push((t,quat));

    }

    ground_truths
}



pub fn load_intrinsics_as_pinhole(file_path: &Path, invert_focal_lengths: bool) -> Pinhole {
    let file = File::open(file_path).expect("load_intrinsics failed");
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).unwrap();
    let values = contents.trim().split(|c| !char::is_numeric(c)).map(|s| (s.len(),s.parse::<f64>())).filter(|(_,option)|  option.is_ok()).map(|(len,option)| (len,option.unwrap())).collect::<Vec<(usize,Float)>>();

    let fx = values[0].1 + values[1].1/(10f64.powi(values[1].0 as i32));
    let fy = values[6].1 +values[7].1/(10f64.powi(values[7].0 as i32));

    let cx = values[3].1;
    let cy = values[8].1;

    Pinhole::new(fx, fy, cx, cy, invert_focal_lengths)

}