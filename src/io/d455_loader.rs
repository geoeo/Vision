extern crate nalgebra as na;
extern crate image as image_rs;

use std::path::Path;
use std::fs::File;
use std::io::{BufReader,Read,BufRead};


use crate::Float;
use crate::io::{loading_parameters::LoadingParameters,loaded_data::LoadedData, parse_to_float, closest_ts_index};
use crate::image::{Image};
use crate::camera::pinhole::Pinhole;
use crate::imu::Imu;
use crate::io::{load_image_as_gray,load_depth_image_from_csv};



pub fn load(root_path: &str, parameters: &LoadingParameters) -> LoadedData {
    let depth_image_format = "csv";
    let color_image_format = "png";
    let text_format = "txt";

    let depth_image_folder = format!("{}/{}",root_path,"depth");
    let color_image_folder = format!("{}/{}",root_path,"rgb");

    let pinhole_camera = Pinhole::new(381.963043212891, 381.700378417969, 320.757202148438, 245.415313720703, parameters.invert_focal_lengths);
    let (rgb_ts,rgb_ts_string) = load_timestamps(Path::new(&color_image_folder),&color_image_format,false, true);
    let (depth_ts,depth_ts_string) = load_timestamps(Path::new(&depth_image_folder),&depth_image_format,false, true);

    let source_rgb_indices = (parameters.starting_index..parameters.starting_index+parameters.count).step_by(parameters.step);
    let target_rgb_indices = source_rgb_indices.clone().map(|x| x + parameters.step); //TODO: check out of range

    let source_rgb_ts = source_rgb_indices.clone().map(|x| rgb_ts[x]).collect::<Vec<Float>>();
    let target_rgb_ts = target_rgb_indices.clone().map(|x| rgb_ts[x]).collect::<Vec<Float>>();

    let source_depth_indices = source_rgb_ts.iter().map(|&x| closest_ts_index(x, &depth_ts)).collect::<Vec<usize>>();
    let target_depth_indices = target_rgb_ts.iter().map(|&x| closest_ts_index(x, &depth_ts)).collect::<Vec<usize>>(); //TODO: check out of range

    LoadedData {
        source_gray_images: source_rgb_indices.map(|i| {
            let color_source_image_path = format!("{}/{}",color_image_folder,rgb_ts_string[i]);
            load_image_as_gray(&Path::new(&color_source_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        source_depth_images: source_depth_indices.iter().map(|&i| {
            let depth_source_image_path = format!("{}/{}",depth_image_folder,depth_ts_string[i]);
            load_depth_image_from_csv(&Path::new(&depth_source_image_path), parameters.negate_depth_values, parameters.invert_y,640,480,1.0,false,parameters.set_default_depth) //TODO: pass into
        }).collect::<Vec<Image>>(),
        target_gray_images: target_rgb_indices.map(|i| {
            let color_target_image_path = format!("{}/{}",color_image_folder,rgb_ts_string[i]);
            load_image_as_gray(&Path::new(&color_target_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        target_depth_images:  target_depth_indices.iter().map(|&i| {
            let depth_target_image_path = format!("{}/{}",depth_image_folder,depth_ts_string[i]);
            load_depth_image_from_csv(&Path::new(&depth_target_image_path), parameters.negate_depth_values, parameters.invert_y,640,480,1.0,false,parameters.set_default_depth) //TODO: pass into
        }).collect::<Vec<Image>>(),
        pinhole_camera,
        target_gt_poses: None,
        source_gt_poses: None
    }
}

pub fn load_imu(root_path: &str) -> Vec<Imu> {

    let text_format = "txt";
    let delimeters = &[' ','[',']'][..];

    let linear_acc_folder = format!("{}/{}",root_path,"linear_acc");
    let rotational_vel_folder = format!("{}/{}",root_path,"angular_vel");

    let (linear_acc_ts,linear_acc_ts_string) = load_timestamps(Path::new(&linear_acc_folder), &text_format, false, true);
    let (rotational_vel_ts,rotational_vel_ts_string) = load_timestamps(Path::new(&rotational_vel_folder), &text_format, false, true);

    let linear_acc_file_paths = linear_acc_ts_string.iter().map(|x| format!("{}/{}",linear_acc_folder,x)).collect::<Vec<String>>();
    let rotational_vel_file_paths = rotational_vel_ts_string.iter().map(|x| format!("{}/{}",rotational_vel_folder,x)).collect::<Vec<String>>();

    let linear_acc_vec = linear_acc_file_paths.iter().map(|x| load_measurement(Path::new(x),delimeters)).collect::<Vec<Vec<Float>>>();
    let rotational_vel_vec = rotational_vel_file_paths.iter().map(|x| load_measurement(Path::new(x),delimeters)).collect::<Vec<Vec<Float>>>();

    let closest_rotational_vel_indices = linear_acc_ts.iter().map(|&x| closest_ts_index(x, &rotational_vel_ts)).collect::<Vec<usize>>();
    let closest_rotational_vec = closest_rotational_vel_indices.iter().map(|&x| rotational_vel_vec[x].clone()).collect::<Vec<Vec<Float>>>();

    assert_eq!(closest_rotational_vec.len(),linear_acc_vec.len());

    linear_acc_vec.iter().zip(closest_rotational_vec.iter()).map(|(l_a,r_v)| Imu::new(l_a,r_v)).collect::<Vec<Imu>>()
}

// TODO: This can be put outside of loader
fn load_timestamps(dir_path: &Path, file_format: &str, negate_value: bool, sort_result: bool)-> (Vec<Float>,Vec<String>) {
    let mut ts_string = Vec::<(Float,String)>::new();
    let mut timestamps = Vec::<Float>::new();
    let mut timestamps_string = Vec::<String>::new();

    if dir_path.is_dir() {
        for entry_result in std::fs::read_dir(dir_path).unwrap() {
            let entry = entry_result.unwrap();
            if entry.path().is_file() {
                let full_file_name = entry.file_name().into_string().unwrap();
                let split = full_file_name.split('.').collect::<Vec<&str>>();
                let file_type = split[split.len()-1];
                if file_type.eq(file_format) {
                    let mut ts_parts = Vec::<&str>::with_capacity(split.len()-1);
                    for i in 0..split.len()-1 {
                        ts_parts.push(split[i]);
                    }

                    let first = ts_parts[0];
                    let prefix_split = first.split('_').collect::<Vec<&str>>();
                    ts_parts[0] = prefix_split[prefix_split.len()-1];
                    let ts_str = ts_parts.join(".");

                    ts_string.push((parse_to_float(ts_str.as_str(), negate_value),full_file_name));
                }

            }
        }

    }

    if sort_result {
        ts_string.sort_unstable_by(|(a,_),(b,_)| a.partial_cmp(b).unwrap());
    }

    for (v,s) in ts_string {
        timestamps.push(v);
        timestamps_string.push(s);
    }

    assert_ne!(timestamps.len(),0);
    assert_eq!(timestamps.len(),timestamps_string.len());

    (timestamps,timestamps_string)
}

fn load_measurement(file_path: &Path, delimeters: &[char]) -> Vec<Float> {
    let file = File::open(file_path).expect(format!("Could not open: {}", file_path.display()).as_str());
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let line = lines.next().unwrap();
    let contents = line.unwrap();
    let values = contents.trim().split(delimeters).filter(|&x| x!="").collect::<Vec<&str>>();
    assert_eq!(values.len(),3);

    values.iter().map(|x| parse_to_float(x, false)).collect::<Vec<Float>>()
}