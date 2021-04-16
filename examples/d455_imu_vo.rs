extern crate nalgebra as na;
extern crate vision;

use na::{Vector4,Matrix4, Vector3, UnitQuaternion};
use std::boxed::Box;
use vision::io::{loading_parameters::LoadingParameters,d455_loader};
use vision::pyramid::gd::{GDPyramid,gd_octave::GDOctave, build_rgbd_pyramid,gd_runtime_parameters::GDRuntimeParameters};
use vision::odometry::imu_odometry::{pre_integration,solver::run_trajectory};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::Float;
use vision::{numerics,numerics::loss};
use vision::visualize::plot;

fn main() {


    let dataset_name = "simple_trans_imu";

    let root_path = format!("C:/Users/Marc/Workspace/Datasets/D455/{}",dataset_name);
    let out_folder = "C:/Users/Marc/Workspace/Rust/Vision/output";

    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :20,
        negate_depth_values :false,
        invert_focal_lengths :false,
        invert_y :true,
        set_default_depth: false,
        gt_alignment_rot:UnitQuaternion::identity()
    };

    let loaded_data = d455_loader::load_camera(&root_path, &loading_parameters);

    let source_gray_images = loaded_data.source_gray_images;
    let source_depth_images = loaded_data.source_depth_images;
    let target_gray_images = loaded_data.target_gray_images;
    let target_depth_images = loaded_data.target_depth_images;
    let intensity_cam = loaded_data.intensity_camera;
    let depth_cam = loaded_data.depth_camera;

    println!("{:?}",loaded_data.intensity_camera.projection);

    let pyramid_parameters = GDRuntimeParameters{
        sigma: 0.5,
        use_blur: true,
        blur_radius: 1.0,
        octave_count: 3,
        min_image_dimensions: (50,50),
        invert_grad_x : true,
        invert_grad_y : true,
        blur_grad_x : false,
        blur_grad_y: false,
        normalize_gray: true,
        normalize_gradients: false
    };

    let vo_parameters = RuntimeParameters{
        max_iterations: vec![800;3],
        eps: vec!(1e-3,1e-3,1e-3),
        step_sizes: vec!(1e-8,1e-8,1e-8), 
        max_norm_eps: 1e-95,
        delta_eps: 1e-95,
        taus: vec!(1e-3,1e-3,1e-0), 
        lm: true,
        weighting: true,
        debug: false,

        show_octave_result: true,
        loss_function: Box::new(numerics::loss::TrivialLoss {eps: 1e-16})
    };

    let mut se3_est = vec!(Matrix4::<Float>::identity());

    let data_frame = d455_loader::load_data_frame(&root_path, &loading_parameters);
    let imu_data = &data_frame.imu_data_vec;

    se3_est.extend(run_trajectory(&imu_data,&Vector3::<Float>::zeros(),&Vector3::<Float>::zeros(), &Vector3::<Float>::new(0.0,9.81,0.0), &vo_parameters)); 


    // for (i,data) in imu_data.iter().enumerate() {
    //     let imu_delta = pre_integration(data,&Vector3::<Float>::zeros(),&Vector3::<Float>::zeros(), &Vector3::<Float>::new(0.0,9.81,0.0));
    //     let pose = imu_delta.get_pose();
    //     se3_est.push(pose);
    // }
    

    let est_points = numerics::pose::apply_pose_deltas_to_point(Vector4::<Float>::new(0.0,0.0,0.0,1.0), &se3_est);

    let out_file_name = format!("d455_imu_{}.png",dataset_name);

    let title = "d455";
    plot::draw_line_graph_vector3(&est_points.iter().map(|x| Vector3::<Float>::new(x[0],x[1], x[2])).collect::<Vec<Vector3<Float>>>(), out_folder, &out_file_name, &title, &"Translation", &"meters");



}