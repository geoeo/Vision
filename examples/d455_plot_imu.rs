extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;


use na::{Vector3,UnitQuaternion};
use vision::io::{loading_parameters::LoadingParameters,d455_loader};
use vision::visualize::plot;
use vision::Float;

fn main() {

    let dataset_name = "simple_trans_imu";
    let output_folder = "C:/Users/Marc/Workspace/Rust/Vision/output/";
    let root_path = format!("C:/Users/Marc/Workspace/Datasets/D455/{}",dataset_name);
    let imu_data_frame = d455_loader::load_imu(&root_path);

    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :1,
        negate_depth_values :true,
        invert_focal_lengths :true,
        invert_y :true,
        set_default_depth: true,
        gt_alignment_rot:UnitQuaternion::identity()
    };

    let data_frame = d455_loader::load_data_frame(&root_path, &loading_parameters);

    let out_file_name_accelerometer = format!("d455_{}_accelerometer.png",dataset_name);
    let out_file_name_gyro = format!("d455_{}_gyroscope.png",dataset_name);
    //let info = format!("{}_{}_{}",loading_parameters,pyramid_parameters,vo_parameters);


    let title_accelerometer = "accelerometer";
    plot::draw_line_graph_vector3(&imu_data_frame.acceleration_data, &output_folder, &out_file_name_accelerometer,&title_accelerometer, &title_accelerometer, &"g(10 m/s²)");
    
    let title_gyro = "gyroscore";
    plot::draw_line_graph_vector3(&imu_data_frame.gyro_data, &output_folder, &out_file_name_gyro,&title_gyro, &title_gyro, &"°/s");


}