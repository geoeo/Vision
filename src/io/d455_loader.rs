extern crate nalgebra as na;
extern crate image as image_rs;

use std::path::Path;
use std::fs::File;
use std::io::{BufReader,Read,BufRead};


use na::{U3,U1,Vector3,Matrix4};
use crate::Float;
use crate::io::{image_loading_parameters::ImageLoadingParameters,imu_loading_parameters::ImuLoadingParameters, parse_to_float, closest_ts_index};
use crate::image::{Image};
use crate::sensors::camera::{camera_data_frame::CameraDataFrame,pinhole::Pinhole};
use crate::sensors::imu::{imu_data_frame::ImuDataFrame,bmi005};
use crate::sensors::DataFrame;
use crate::io::{load_image_as_gray,load_depth_image_from_csv};



pub fn load_camera(root_path: &str, parameters: &ImageLoadingParameters) -> CameraDataFrame {
    let depth_image_format = "csv";
    let color_image_format = "png";
    let text_format = "txt";
    let depth_image_folder = format!("{}/{}",root_path,"depth");
    let color_image_folder = format!("{}/{}",root_path,"rgb");

    //let intensity_camera = Pinhole::new(381.963043212891, 381.700378417969, 320.757202148438, 245.415313720703, parameters.invert_focal_lengths);
    let intensity_camera = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, parameters.invert_focal_lengths);
    let depth_camera = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, parameters.invert_focal_lengths);

    let (rgb_ts,rgb_ts_string) = load_timestamps(Path::new(&color_image_folder),&color_image_format,false, true,1e-3);
    let (depth_ts,depth_ts_string) = load_timestamps(Path::new(&depth_image_folder),&depth_image_format,false, true,1e-3);

    let source_rgb_indices = (parameters.starting_index..parameters.starting_index+parameters.count).step_by(parameters.step);
    let target_rgb_indices = source_rgb_indices.clone().map(|x| x + parameters.step); //TODO: check out of range

    let source_rgb_ts = source_rgb_indices.clone().map(|x| rgb_ts[x]).collect::<Vec<Float>>();
    let target_rgb_ts = target_rgb_indices.clone().map(|x| rgb_ts[x]).collect::<Vec<Float>>();

    let source_depth_indices = source_rgb_ts.iter().map(|&x| closest_ts_index(x, &depth_ts)).collect::<Vec<usize>>();
    let target_depth_indices = target_rgb_ts.iter().map(|&x| closest_ts_index(x, &depth_ts)).collect::<Vec<usize>>(); //TODO: check out of range

    let mut depth_camera_transform = Matrix4::<Float>::identity();
    depth_camera_transform.fixed_slice_mut::<U3,U1>(0,3).copy_from(&Vector3::<Float>::new(-0.059157,0.0,-0.000390));

    CameraDataFrame {
        source_timestamps: source_rgb_ts,
        target_timestamps: target_rgb_ts,
        source_gray_images: source_rgb_indices.map(|i| {
            let color_source_image_path = format!("{}/{}",color_image_folder,rgb_ts_string[i]);
            load_image_as_gray(&Path::new(&color_source_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        source_depth_images: source_depth_indices.iter().map(|&i| {
            let depth_source_image_path = format!("{}/{}",depth_image_folder,depth_ts_string[i]);
            load_depth_image_from_csv(&Path::new(&depth_source_image_path), parameters.negate_depth_values, parameters.invert_y,640,480,1.0,false,parameters.set_default_depth, &Some((&depth_camera_transform,&depth_camera))) //TODO: pass into
        }).collect::<Vec<Image>>(),
        target_gray_images: target_rgb_indices.map(|i| {
            let color_target_image_path = format!("{}/{}",color_image_folder,rgb_ts_string[i]);
            load_image_as_gray(&Path::new(&color_target_image_path), false, parameters.invert_y)
        }).collect::<Vec<Image>>(),
        target_depth_images:  target_depth_indices.iter().map(|&i| {
            let depth_target_image_path = format!("{}/{}",depth_image_folder,depth_ts_string[i]);
            load_depth_image_from_csv(&Path::new(&depth_target_image_path), parameters.negate_depth_values, parameters.invert_y,640,480,1.0,false,parameters.set_default_depth, &Some((&depth_camera_transform,&depth_camera))) //TODO: pass into
        }).collect::<Vec<Image>>(),
        intensity_camera,
        depth_camera,
        target_gt_poses: None,
        source_gt_poses: None
    }
}

//TODO: make loading paramters for IMU
pub fn load_imu(root_path: &str, imu_loading_parameters: &ImuLoadingParameters) -> ImuDataFrame {

    let text_format = "txt";
    let delimeters = &[' ','[',']'][..];

    let linear_acc_folder = format!("{}/{}",root_path,"linear_acc");
    let rotational_vel_folder = format!("{}/{}",root_path,"angular_vel");

    let (linear_acc_ts,linear_acc_ts_string) = load_timestamps(Path::new(&linear_acc_folder), &text_format, false, true, 1e-3);
    let (rotational_vel_ts,rotational_vel_ts_string) = load_timestamps(Path::new(&rotational_vel_folder), &text_format, false, true,1e-3);

    let linear_acc_file_paths = linear_acc_ts_string.iter().map(|x| format!("{}/{}",linear_acc_folder,x)).collect::<Vec<String>>();
    let rotational_vel_file_paths = rotational_vel_ts_string.iter().map(|x| format!("{}/{}",rotational_vel_folder,x)).collect::<Vec<String>>();

    let linear_acc_vec = linear_acc_file_paths.iter().map(|x| load_measurement(Path::new(x),delimeters, false,false,false)).collect::<Vec<Vector3<Float>>>();
    let rotational_vel_vec = rotational_vel_file_paths.iter().map(|x| load_measurement(Path::new(x),delimeters, false,false,false)).collect::<Vec<Vector3<Float>>>();

    bmi005::new_dataframe_from_data(rotational_vel_vec,rotational_vel_ts,linear_acc_vec,linear_acc_ts)
}

pub fn load_data_frame(root_path: &str, parameters: &ImageLoadingParameters, imu_loading_parameters: &ImuLoadingParameters) -> DataFrame {
    let camera_data = load_camera(root_path,parameters);
    let imu_data = load_imu(root_path,imu_loading_parameters);

    DataFrame::new(camera_data, imu_data)
}

// TODO: This can be put outside of loader
fn load_timestamps(dir_path: &Path, file_format: &str, negate_value: bool, sort_result: bool, ts_scaling: Float)-> (Vec<Float>,Vec<String>) {
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

                    ts_string.push((parse_to_float(ts_str.as_str(), negate_value)*ts_scaling,full_file_name));
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

fn load_measurement(file_path: &Path, delimeters: &[char], invert_x: bool, invert_y: bool, invert_z: bool) -> Vector3<Float> {
    let file = File::open(file_path).expect(format!("Could not open: {}", file_path.display()).as_str());
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let line = lines.next().unwrap();
    let contents = line.unwrap();
    let values = contents.trim().split(delimeters).filter(|&x| x!="").collect::<Vec<&str>>();
    assert_eq!(values.len(),3);

    Vector3::<Float>::new(parse_to_float(values[0], invert_x),parse_to_float(values[1], invert_y),parse_to_float(values[2], invert_z))
}