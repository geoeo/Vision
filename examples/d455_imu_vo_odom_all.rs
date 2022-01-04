extern crate nalgebra as na;
extern crate vision;

use na::{Point3, UnitQuaternion, Vector3, Isometry3};
use std::boxed::Box;
use vision::io::{
    d455_loader, image_loading_parameters::ImageLoadingParameters,
    imu_loading_parameters::ImuLoadingParameters,
};
use vision::odometry::imu_odometry::{pre_integration};
use vision::odometry::joint::run_trajectory;
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::image::pyramid::gd::{
    build_rgbd_pyramid, gd_octave::GDOctave, gd_runtime_parameters::GDRuntimeParameters, GDPyramid,
};
use vision::numerics::{loss,weighting};
use vision::visualize::plot;
use vision::Float;
use vision::numerics;

fn main() {
    let dataset_name = "simple_trans_imu";
    //let dataset_name = "x";

    let root_path = format!("D:/Workspace/Datasets/D455/{}", dataset_name);
    let out_folder = "D:/Workspace/Rust/Vision/output";

    let image_loading_parameters = ImageLoadingParameters {
        starting_index: 5,
        step: 1,
        count: 50,
        image_height: 480,
        image_width: 640,
        negate_depth_values: true,
        invert_focal_lengths: true,
        invert_y: true,
        set_default_depth: true,
        gt_alignment_rot: UnitQuaternion::identity(),
    };

    let imu_loading_parameters = ImuLoadingParameters {
        convert_to_cam_coords: true, //TODO: check this
    };

    let loaded_data = d455_loader::load_data_frame(&root_path, &image_loading_parameters, &imu_loading_parameters);
    let camera_data = loaded_data.camera_data;

    let pyramid_parameters = GDRuntimeParameters{
        pyramid_scale: 2.0,
        sigma: 0.01,
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

    let source_pyramids = camera_data.source_gray_images.into_iter().zip(camera_data.source_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<GDPyramid<GDOctave>>>();
    let target_pyramids = camera_data.target_gray_images.into_iter().zip(camera_data.target_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<GDPyramid<GDOctave>>>();

    let vo_parameters = RuntimeParameters {
        pyramid_scale: pyramid_parameters.pyramid_scale,
        max_iterations: vec![100; 3],
        eps: vec![1e-3;3],
        step_sizes: vec![1e-3;3],
        max_norm_eps: 1e-5,
        delta_eps: 1e-5,
        taus: vec!(1e-6,1e-3,1e-3),
        lm: true,
        weighting: true,
        debug: false,

        show_octave_result: true,
        loss_function: Box::new(loss::CauchyLoss { eps: 1e-16, approximate_gauss_newton_matrices: true}),
        intensity_weighting_function:  Box::new(weighting::HuberWeightForPos {delta: 1.0})
    };

    let mut se3_est = vec![Isometry3::<Float>::identity()];
    let mut se3_est_imu = vec![Isometry3::<Float>::identity()];
    let mut se3_preintegration_est = vec![Isometry3::<Float>::identity()];

    let data_frame = d455_loader::load_data_frame(
        &root_path,
        &image_loading_parameters,
        &imu_loading_parameters,
    );
    let imu_data = &data_frame.imu_data_vec;

    se3_est.extend(run_trajectory(
        &source_pyramids,
        &target_pyramids,
        &camera_data.intensity_camera,
        &camera_data.depth_camera,
        &imu_data,
        &Vector3::<Float>::new(0.0, 9.81, 0.0),
        &vo_parameters
    ));

    se3_est_imu.extend(vision::odometry::imu_odometry::solver::run_trajectory(        
        &imu_data,
        &Vector3::<Float>::new(0.0, 9.81, 0.0),
        &vo_parameters));

    let est_points = numerics::pose::apply_pose_deltas_to_point(
        Point3::<Float>::new(0.0, 0.0, 0.0),
        &se3_est,
    );

    let imu_est_points = numerics::pose::apply_pose_deltas_to_point(
        Point3::<Float>::new(0.0, 0.0, 0.0),
        &se3_est_imu,
    );

    for (i, data) in imu_data.iter().enumerate() {
        let (imu_delta, _,_) = pre_integration(
            data,
            &Vector3::<Float>::new(0.0, 9.81, 0.0),
        );
        let pose = imu_delta.get_pose();
        se3_preintegration_est.push(pose);
    }
    let preintegration_points = numerics::pose::apply_pose_deltas_to_point(
        Point3::<Float>::new(0.0, 0.0, 0.0),
        &se3_preintegration_est,
    );

    let out_file_name = format!("d455_all_{}_{}_{}.png", dataset_name, vo_parameters,imu_loading_parameters);

    let title = "solver";
    plot::draw_line_graph_three_vector3(
        &est_points,
            &"joint estimated",
        &preintegration_points,
            &"preintegration",
        &imu_est_points,
            &"imu only",
        out_folder,
        &out_file_name,
        &title,
        &"Translation",
        &"meters",
    );


}
