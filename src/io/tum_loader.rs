extern crate nalgebra as na;
extern crate image as image_rs;

use na::{Vector3, Quaternion, UnitQuaternion};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader,BufRead};

use crate::Float;
use crate::io::{image_loading_parameters::ImageLoadingParameters, parse_to_float, closest_ts_index};
use crate::image::Image;
use crate::sensors::camera::perspective::Perspective;
use crate::io::{load_image_as_gray,load_depth_image,camera_data_frame::CameraDataFrame};

#[repr(u8)]
pub enum Dataset {
    FR1,
    FR2,
    FR3,
}


pub fn load(root_path: &str, parameters: &ImageLoadingParameters, dataset: &Dataset) -> CameraDataFrame {
    let rgb_ts_names = "rgb";
    let depth_ts_names = "depth";
    let ground_truths = "groundtruth";

    let depth_image_format = "png";
    let color_image_format = "png";
    let text_format = "txt";

    let depth_image_folder = format!("{}/{}",root_path,"depth");
    let color_image_folder = format!("{}/{}",root_path,"rgb");
    let rgb_ts_name_path = format!("{}/{}.{}",root_path,rgb_ts_names, text_format);
    let depth_ts_name_path = format!("{}/{}.{}",root_path,depth_ts_names, text_format);
    let ground_truths_path = format!("{}/{}.{}",root_path,ground_truths, text_format);

    let rgb_ts = load_timestamps(&Path::new(&rgb_ts_name_path));
    let depth_ts = load_timestamps(&Path::new(&depth_ts_name_path));
    let (gt_timestamps,ground_truths) = load_ground_truths_and_timestamps(&Path::new(&ground_truths_path), &parameters.gt_alignment_rot);

    let intensity_camera = load_intrinsics_as_pinhole(dataset, parameters.invert_focal_lengths);
    let depth_camera = intensity_camera.clone();

    let source_rgb_indices = (parameters.starting_index..parameters.starting_index+parameters.count).step_by(parameters.step);
    let target_rgb_indices = source_rgb_indices.clone().map(|x| x + parameters.step); //TODO: check out of range

    let source_rgb_ts = source_rgb_indices.map(|x| rgb_ts[x]).collect::<Vec<Float>>();
    let target_rgb_ts = target_rgb_indices.map(|x| rgb_ts[x]).collect::<Vec<Float>>();

    let source_depth_indices = source_rgb_ts.iter().map(|&x| closest_ts_index(x, &depth_ts)).collect::<Vec<usize>>();
    let target_depth_indices = target_rgb_ts.iter().map(|&x| closest_ts_index(x, &depth_ts)).collect::<Vec<usize>>(); //TODO: check out of range

    let source_depth_ts = source_depth_indices.iter().map(|&x| depth_ts[x]).collect::<Vec<Float>>();
    let target_depth_ts = target_depth_indices.iter().map(|&x| depth_ts[x]).collect::<Vec<Float>>();

    let source_gt_indices = source_rgb_ts.iter().map(|&x| closest_ts_index(x, &gt_timestamps)).collect::<Vec<usize>>();
    let target_gt_indices = target_rgb_ts.iter().map(|&x| closest_ts_index(x, &gt_timestamps)).collect::<Vec<usize>>(); //TODO: check out of range

    CameraDataFrame {
        source_timestamps: source_rgb_ts.clone(),
        target_timestamps: target_rgb_ts.clone(),
        source_gray_images: source_rgb_ts.iter().map(|s| {
            let color_source_image_path = format!("{}/{:.6}.{}",color_image_folder,s, color_image_format);
            load_image_as_gray(&Path::new(&color_source_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        source_depth_images: source_depth_ts.iter().map(|s| {
            let depth_source_image_path = format!("{}/{:.6}.{}",depth_image_folder,s, depth_image_format);
            load_depth_image(&Path::new(&depth_source_image_path),parameters.negate_depth_values, true,  5000.0,parameters.set_default_depth)
        }).collect::<Vec<Image>>(),
        target_gray_images: target_rgb_ts.iter().map(|t| {
            let color_target_image_path = format!("{}/{:.6}.{}",color_image_folder,t, color_image_format);
            load_image_as_gray(&Path::new(&color_target_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        target_depth_images:  target_depth_ts.iter().map(|t| {
            let depth_target_image_path = format!("{}/{:.6}.{}",depth_image_folder,t, depth_image_format);
            load_depth_image(&Path::new(&depth_target_image_path), parameters.negate_depth_values, true, 5000.0, parameters.set_default_depth)
        }).collect::<Vec<Image>>(),
        intensity_camera,
        depth_camera,
        target_gt_poses: Some(target_gt_indices.iter().map(|&t| {
            ground_truths[t]
        }).collect::<Vec<(Vector3<Float>,Quaternion<Float>)>>()),
        source_gt_poses: Some(source_gt_indices.iter().map(|&s| {
            ground_truths[s]
        }).collect::<Vec<(Vector3<Float>,Quaternion<Float>)>>())
    }
}



fn load_timestamps(file_path: &Path)-> Vec<Float> {
    let file = File::open(file_path).expect("load_timestamps failed");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    let mut timestamps = Vec::<Float>::new();

    for line in lines {
        let contents = line.unwrap();
        if contents.starts_with('#') {
            continue;
        }
        let values = contents.trim().split(" ").collect::<Vec<&str>>();
        let ts = parse_to_float(values[0],false);
        timestamps.push(ts);
    }

    timestamps

}



pub fn load_ground_truths_and_timestamps(file_path: &Path, gt_alignment_rot: &UnitQuaternion<Float>) -> (Vec<Float>,Vec<(Vector3<Float>,Quaternion<Float>)>) {
    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    let mut ground_truths = Vec::<(Vector3<Float>,Quaternion<Float>)>::new();
    let mut timestamps = Vec::<Float>::new();

    for line in lines {
        let contents = line.unwrap();
        if contents.starts_with('#') {
            continue;
        }
        let values = contents.trim().split(" ").map(|x| parse_to_float(x,false)).collect::<Vec<Float>>();
        let ts = values[0];
        let tx = values[1];
        let ty = values[2];
        let tz = values[3];

        let qx =  values[4];
        let qy =  values[5];
        let qz =  values[6];
        let qw =  values[7];

        let translation = gt_alignment_rot*Vector3::<Float>::new(tx,ty,tz);
        let quaternion = gt_alignment_rot.quaternion()*Quaternion::<Float>::new(qw,qx,qy,qz);

        timestamps.push(ts);
        ground_truths.push((translation,quaternion));


    }

    (timestamps,ground_truths)
}

pub fn load_intrinsics_as_pinhole(dataset: &Dataset, invert_focal_lengths: bool) -> Perspective<Float> {
    match dataset {
        Dataset::FR1 => Perspective::new(517.3, 516.5, 318.6, 255.3, 0.0, invert_focal_lengths),
        Dataset::FR2 => Perspective::new(520.9, 521.0, 325.1, 249.7, 0.0, invert_focal_lengths),
        Dataset::FR3 => Perspective::new(535.4, 539.2, 320.1, 247.6, 0.0, invert_focal_lengths)
    }
}