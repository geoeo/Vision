extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;


use na::Vector3;
use std::path::Path;
use vision::io::{load_depth_image_from_csv,load_image_as_gray,d455_loader};
use vision::visualize::plot;
use vision::Float;

fn main() {

    let dataset_name = "simple_trans_imu";
    let output_folder = "C:/Users/Marc/Workspace/Rust/Vision/output/";
    let imu_data_frame = d455_loader::load_imu(&format!("C:/Users/Marc/Workspace/Datasets/D455/{}",dataset_name));

    let out_file_name_accelerometer = format!("d455_{}_accelerometer.png",dataset_name);
    let out_file_name_gyro = format!("d455_{}_gyroscope.png",dataset_name);
    //let info = format!("{}_{}_{}",loading_parameters,pyramid_parameters,vo_parameters);

    let accelerometer_data = imu_data_frame.imu_data.iter().map(|&x| x.accelerometer).collect::<Vec<Vector3<Float>>>();
    let gyroscope_data = imu_data_frame.imu_data.iter().map(|&x| x.gyro).collect::<Vec<Vector3<Float>>>();

    let title_accelerometer = "accelerometer";
    plot::draw_line_graph_vector3(&accelerometer_data, &output_folder, &out_file_name_accelerometer,&title_accelerometer, &title_accelerometer, &"g(10 m/s²)");
    
    let title_gyro = "gyroscore";
    plot::draw_line_graph_vector3(&gyroscope_data, &output_folder, &out_file_name_gyro,&title_gyro, &title_gyro, &"°/s");


}