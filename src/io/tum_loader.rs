extern crate nalgebra as na;
extern crate image as image_rs;

use na::{Vector3, Quaternion, UnitQuaternion};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader,BufRead};

use crate::{Float,float};
use crate::io::{loading_parameters::LoadingParameters,loaded_data::LoadedData, parse_to_float};
use crate::image::{Image};
use crate::camera::pinhole::Pinhole;

#[repr(u8)]
pub enum Dataset {
    FR1,
    FR2,
    FR3,
}


pub fn load(root_path: &str, parameters: &LoadingParameters, dataset: &Dataset) -> LoadedData {
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
    let (gt_timestamps,ground_truths) = load_timestamps_ground_truths(&Path::new(&ground_truths_path), &parameters.gt_alignment_rot);

    let pinhole_camera = load_intrinsics_as_pinhole(dataset, parameters.invert_focal_lengths);

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

    LoadedData {
        source_gray_images: source_rgb_ts.iter().map(|s| {
            let color_source_image_path = format!("{}/{:.6}.{}",color_image_folder,s, color_image_format);
            load_image_as_gray(&Path::new(&color_source_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        source_depth_images: source_depth_ts.iter().map(|s| {
            let depth_source_image_path = format!("{}/{:.6}.{}",depth_image_folder,s, depth_image_format);
            load_depth_image(&Path::new(&depth_source_image_path),parameters.negate_values, true)
        }).collect::<Vec<Image>>(),
        target_gray_images: target_rgb_ts.iter().map(|t| {
            let color_target_image_path = format!("{}/{:.6}.{}",color_image_folder,t, color_image_format);
            load_image_as_gray(&Path::new(&color_target_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        target_depth_images:  target_depth_ts.iter().map(|t| {
            let depth_target_image_path = format!("{}/{:.6}.{}",depth_image_folder,t, depth_image_format);
            load_depth_image(&Path::new(&depth_target_image_path), parameters.negate_values, true)
        }).collect::<Vec<Image>>(),
        source_gt_poses: source_gt_indices.iter().map(|&s| {
            ground_truths[s]
        }).collect::<Vec<(Vector3<Float>,Quaternion<Float>)>>(),
        target_gt_poses: target_gt_indices.iter().map(|&t| {
            ground_truths[t]
        }).collect::<Vec<(Vector3<Float>,Quaternion<Float>)>>(),
        pinhole_camera
    }
}

fn closest_ts_index(ts: Float, list: &Vec<Float>) -> usize {
    let mut min_delta = float::MAX;
    let mut min_idx = list.len()-1;

    for (idx, target_ts) in list.iter().enumerate() {
        let delta = (ts-target_ts).abs();
        
        if delta < min_delta {
            min_delta = delta;
            min_idx = idx;
        } else {
            break;
        }    
    }

    min_idx
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



pub fn load_timestamps_ground_truths(file_path: &Path, gt_alignment_rot: &UnitQuaternion<Float>) -> (Vec<Float>,Vec<(Vector3<Float>,Quaternion<Float>)>) {
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

pub fn load_depth_image(file_path: &Path, negate_values: bool, invert_y: bool) -> Image {
    let depth_image = image_rs::open(&Path::new(&file_path)).expect("load_image failed").to_luma16();
    let mut image = Image::from_depth_image(&depth_image,negate_values,invert_y);
    image.buffer /= 5000.0;
    let extrema = match invert_y {
        true => image.buffer.min(),
        false => image.buffer.max()
    };
    for r in 0..image.buffer.nrows(){
        for c in 0..image.buffer.ncols(){
            if image.buffer[(r,c)] == 0.0 {
                image.buffer[(r,c)] = extrema;
            }
        }
    }
    image
}

pub fn load_image_as_gray(file_path: &Path, normalize: bool, invert_y: bool) -> Image {
    let gray_image = image_rs::open(&Path::new(&file_path)).expect("load_image failed").to_luma8();
    Image::from_gray_image(&gray_image,normalize,invert_y)
}


pub fn load_intrinsics_as_pinhole(dataset: &Dataset, invert_focal_lengths: bool) -> Pinhole {
    match dataset {
        Dataset::FR1 => Pinhole::new(517.3, 516.5, 318.6, 255.3, invert_focal_lengths),
        Dataset::FR2 => Pinhole::new(520.9, 521.0, 325.1, 249.7, invert_focal_lengths),
        Dataset::FR3 => Pinhole::new(535.4, 539.2, 320.1, 247.6, invert_focal_lengths)
    }
}