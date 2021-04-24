extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;


use na::{Vector3,UnitQuaternion};
use vision::io::{imu_loading_parameters::ImuLoadingParameters,d455_loader};
use vision::visualize::plot;
use vision::{float,Float};

fn main() {

    let dataset_name = "x";
    let output_folder = "D:/Workspace/Rust/Vision/output/";
    let root_path = format!("D:/Workspace/Datasets/D455/{}",dataset_name);


    let imu_loading_parameters = ImuLoadingParameters {
        convert_to_cam_coords: true
    };


    let imu_data_frame = d455_loader::load_imu(&root_path, &imu_loading_parameters);


    let title_accelerometer = "accelerometer";
    let title_gyro = "gyroscore";


    let out_file_name_accelerometer = format!("d455_{}_accelerometer.png",dataset_name);
    let out_file_name_gyro = format!("d455_{}_gyroscope.png",dataset_name);
    plot::draw_line_graph_vector3(&imu_data_frame.acceleration_data, &output_folder, &out_file_name_accelerometer,&title_accelerometer, &title_accelerometer, &"m/s²");
    plot::draw_line_graph_vector3(&imu_data_frame.gyro_data, &output_folder, &out_file_name_gyro,&title_gyro, &title_gyro, &"rad/s");

    

    // let loading_parameters = ImageLoadingParameters {
    //     starting_index: 10,
    //     step :1,
    //     count :310,
    //     negate_depth_values :true,
    //     invert_focal_lengths :true,
    //     invert_y :true,
    //     set_default_depth: true,
    //     gt_alignment_rot:UnitQuaternion::identity()
    // };
    
    // let data_frame = d455_loader::load_data_frame(&root_path, &loading_parameters);
    // let cam_frame = 0;
    // let out_file_name_accelerometer_cam_frame = format!("d455_{}_accelerometer_cam_{}.png",dataset_name, cam_frame);
    // let out_file_name_gyro_cam_frame = format!("d455_{}_gyroscope_cam_{}.png",dataset_name,cam_frame);

    // plot::draw_line_graph_vector3(&data_frame.imu_data_vec[cam_frame].acceleration_data, &output_folder, &out_file_name_accelerometer_cam_frame,&title_accelerometer, &title_accelerometer, &"m/s²");
    // plot::draw_line_graph_vector3(&data_frame.imu_data_vec[cam_frame].gyro_data, &output_folder, &out_file_name_gyro_cam_frame,&title_gyro, &title_gyro, &"rad/s");


}