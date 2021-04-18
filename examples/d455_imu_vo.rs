extern crate nalgebra as na;
extern crate vision;

use na::{Matrix4, UnitQuaternion, Vector3, Vector4};
use std::boxed::Box;
use vision::io::{
    d455_loader, image_loading_parameters::ImageLoadingParameters,
    imu_loading_parameters::ImuLoadingParameters,
};
use vision::odometry::imu_odometry::{pre_integration, solver::run_trajectory};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::pyramid::gd::{
    build_rgbd_pyramid, gd_octave::GDOctave, gd_runtime_parameters::GDRuntimeParameters, GDPyramid,
};
use vision::visualize::plot;
use vision::{float, Float};
use vision::{numerics, numerics::loss};

fn main() {
    let dataset_name = "simple_trans_imu";

    let root_path = format!("C:/Users/Marc/Workspace/Datasets/D455/{}", dataset_name);
    let out_folder = "C:/Users/Marc/Workspace/Rust/Vision/output";

    let image_loading_parameters = ImageLoadingParameters {
        starting_index: 10,
        step: 1,
        count: 600,
        negate_depth_values: false,
        invert_focal_lengths: false,
        invert_y: true,
        set_default_depth: false,
        gt_alignment_rot: UnitQuaternion::identity(),
    };

    let imu_loading_parameters = ImuLoadingParameters {
        convert_to_cam_coords: true, //TODO: check this
    };

    let vo_parameters = RuntimeParameters {
        max_iterations: vec![800; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e-8],
        max_norm_eps: 1e-100,
        delta_eps: 1e-100,
        taus: vec![1e-12],
        lm: true,
        weighting: true,
        debug: false,

        show_octave_result: true,
        loss_function: Box::new(numerics::loss::SoftOneLoss { eps: 1e-16 }),
    };

    let mut se3_est = vec![Matrix4::<Float>::identity()];
    let mut se3_preintegration_est = vec![Matrix4::<Float>::identity()];

    let data_frame = d455_loader::load_data_frame(
        &root_path,
        &image_loading_parameters,
        &imu_loading_parameters,
    );
    let imu_data = &data_frame.imu_data_vec;

    se3_est.extend(run_trajectory(
        &imu_data,
        &Vector3::<Float>::zeros(),
        &Vector3::<Float>::zeros(),
        &Vector3::<Float>::new(0.0, 9.81, 0.0),
        &vo_parameters,
    ));
    let est_points = numerics::pose::apply_pose_deltas_to_point(
        Vector4::<Float>::new(0.0, 0.0, 0.0, 1.0),
        &se3_est,
    );

    for (i, data) in imu_data.iter().enumerate() {
        let (imu_delta, _) = pre_integration(
            data,
            &Vector3::<Float>::zeros(),
            &Vector3::<Float>::zeros(),
            &Vector3::<Float>::new(0.0, 9.81, 0.0),
        );
        let pose = imu_delta.get_pose();
        se3_preintegration_est.push(pose);
    }
    let preintegration_points = numerics::pose::apply_pose_deltas_to_point(
        Vector4::<Float>::new(0.0, 0.0, 0.0, 1.0),
        &se3_preintegration_est,
    );

    let out_file_name = format!("d455_imu_vo_{}_{}_{}.png", dataset_name, vo_parameters,imu_loading_parameters);

    let title = "solver";
    plot::draw_line_graph_two_vector3(
        &est_points
            .iter()
            .map(|x| Vector3::<Float>::new(x[0], x[1], x[2]))
            .collect::<Vec<Vector3<Float>>>(),
        &preintegration_points
            .iter()
            .map(|x| Vector3::<Float>::new(x[0], x[1], x[2]))
            .collect::<Vec<Vector3<Float>>>(),
        out_folder,
        &out_file_name,
        &title,
        &"Translation",
        &"meters",
    );

    let out_preintegration_file_name = format!("d455_imu_preintegration_{}.png", dataset_name);
    let title = "preintegration";
    // plot::draw_line_graph_vector3(
    //     &preintegration_points
    //         .iter()
    //         .map(|x| Vector3::<Float>::new(x[0], x[1], x[2]))
    //         .collect::<Vec<Vector3<Float>>>(),
    //     out_folder,
    //     &out_preintegration_file_name,
    //     &title,
    //     &"Translation",
    //     &"meters",
    // );
}