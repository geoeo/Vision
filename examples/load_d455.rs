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

    let dataset_name = "simple_trans_imu";
    let root_path = format!("C:/Users/Marc/Workspace/Datasets/D455/{}",dataset_name);

    //TODO: make all parameters
    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :250,
        negate_depth_values :true,
        invert_focal_lengths :true,
        invert_y :true,
        set_default_depth: true,
        gt_alignment_rot:UnitQuaternion::identity()
    };


    //let color_2_image_path = format!("{}{}.{}",color_image_folder,image_2_name, color_image_format);
    //let depth_2_image_path = format!("{}{}.{}",depth_image_folder,image_2_name, depth_image_format);

    //let intrinsics_path = format!("{}{}.{}",intrinsics_folder,intrinsics_name, intrinsics_format);

    //let gray_display = load_image_as_gray(&Path::new(&color_image_path), false, false);
    //let depth_display = load_depth_image_from_csv(&Path::new(&depth_image_path), false, false, 640,480, 1.0, false, true);

    //let depth_2_display = load_depth_image(&Path::new(&depth_2_image_path), false,false, 5000.0);
    //let gray_2_display = load_image_as_gray(&Path::new(&color_2_image_path), false, true);

    //let pinhole_camera = eth_loader::load_intrinsics_as_pinhole(&Path::new(&intrinsics_path), false);

    //let converted_file_out_path = format!("{}{}_out.png",image_out_folder,depth_image_name);
    //let new_image = depth_display.to_image();
    //new_image.save(converted_file_out_path).unwrap();

    //let imu_data = d455_loader::load_imu(&root_path);


    let data_frame = d455_loader::load_data_frame(&root_path, &loading_parameters);

    let camera_data = &data_frame.camera_data;
    let imu_data = &data_frame.imu_data_vec;

    for i in 0..camera_data.source_timestamps.len() {
        let camera_str = format!("Camera: s: {}, t:{}", camera_data.source_timestamps[i], camera_data.target_timestamps[i]);
        println!("{}",camera_str);
        let imu_region = &imu_data[i];

        let a_data_str = format!("# accel: {}", imu_region.acceleration_ts.len());
        println!("{}",a_data_str);
        for j in 0..imu_region.acceleration_ts.len() {
            let imu_ts_str = format!("Accel Imu: ts: {}", imu_region.acceleration_ts[j]);
            println!("{}",imu_ts_str);
        }
        println!("*******");
        let g_data_str = format!("# gyro: {}", imu_region.gyro_ts.len());
        println!("{}",g_data_str);
        for j in 0..imu_region.gyro_ts.len() {
            let imu_ts_str = format!("Gyro Imu: ts: {}", imu_region.gyro_ts[j]);
            println!("{}",imu_ts_str);
        }
        println!("--------------");
    }

    for (i,data) in imu_data.iter().enumerate() {
        let imu_delta = pre_integration(data,&Vector3::<Float>::zeros(),&Vector3::<Float>::zeros(), &Vector3::<Float>::new(0.0,9.81,0.0));
        let pose = imu_delta.get_pose();
        if pose[(0,3)] > pose[(1,3)] {
            println!("{}",i);
            println!("{}",pose);
            println!("--------");
        }

    }






}