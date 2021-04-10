extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;

use na::{Vector3,UnitQuaternion};
use std::path::Path;
use vision::io::{loading_parameters::LoadingParameters,load_depth_image_from_csv,load_image_as_gray,d455_loader};
use vision::odometry::imu_odometry::pre_integration;
use vision::Float;

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

    let dataset_name = "z_2";
    let root_path = format!("C:/Users/Marc/Workspace/Datasets/D455/{}",dataset_name);

    //TODO: make all parameters
    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :10,
        negate_depth_values :false,
        invert_focal_lengths :false,
        invert_y :true,
        set_default_depth: true,
        gt_alignment_rot:UnitQuaternion::identity()
    };



    let converted_file_out_path = format!("{}{}_out.png",image_out_folder,"aligned_depth");

    let data_frame = d455_loader::load_camera(&root_path, &loading_parameters);
    //let data_frame = d455_loader::load_data_frame(&root_path, &loading_parameters);
    let new_image = data_frame.source_depth_images[0].to_image();
    new_image.save(converted_file_out_path).unwrap();

    // let camera_data = &data_frame.camera_data;
    // let imu_data = &data_frame.imu_data_vec;

    // for i in 0..camera_data.source_timestamps.len() {
    //     let camera_str = format!("Camera: s: {}, t:{}", camera_data.source_timestamps[i], camera_data.target_timestamps[i]);
    //     println!("{}",camera_str);
    //     let imu_region = &imu_data[i];

    //     let a_data_str = format!("# accel: {}", imu_region.acceleration_ts.len());
    //     println!("{}",a_data_str);
    //     for j in 0..imu_region.acceleration_ts.len() {
    //         let imu_ts_str = format!("Accel Imu: ts: {}", imu_region.acceleration_ts[j]);
    //         println!("{}",imu_ts_str);
    //     }
    //     println!("*******");
    //     let g_data_str = format!("# gyro: {}", imu_region.gyro_ts.len());
    //     println!("{}",g_data_str);
    //     for j in 0..imu_region.gyro_ts.len() {
    //         let imu_ts_str = format!("Gyro Imu: ts: {}", imu_region.gyro_ts[j]);
    //         println!("{}",imu_ts_str);
    //     }
    //     println!("--------------");
    // }

    // for (i,data) in imu_data.iter().enumerate() {
    //     let imu_delta = pre_integration(data,&Vector3::<Float>::zeros(),&Vector3::<Float>::zeros(), &Vector3::<Float>::new(0.0,9.81,0.0));
    //     let pose = imu_delta.get_pose();
    //         println!("{}",i);
    //         println!("{}",pose);
    //         println!("--------");
    //}






}