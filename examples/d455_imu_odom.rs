extern crate nalgebra as na;
extern crate vision;

use na::{Matrix4, UnitQuaternion, Vector3, Isometry3};
use std::boxed::Box;
use vision::io::{
    d455_loader, image_loading_parameters::ImageLoadingParameters,
    imu_loading_parameters::ImuLoadingParameters,
};
use vision::odometry::imu_odometry::{pre_integration, solver::run_trajectory};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};
use vision::visualize::plot;
use vision::Float;
use vision::numerics;

fn main() {
    let dataset_name = "simple_trans_imu";

    let root_path = format!("D:/Workspace/Datasets/D455/{}", dataset_name);
    let out_folder = "D:/Workspace/Rust/Vision/output";

    let image_loading_parameters = ImageLoadingParameters {
        starting_index: 5,
        step: 1,
        count: 150,
        image_height: 480,
        image_width: 640,
        negate_depth_values: false,
        invert_focal_lengths: false,
        invert_y: true,
        set_default_depth: true,
        gt_alignment_rot: UnitQuaternion::identity(),
    };

    let imu_loading_parameters = ImuLoadingParameters {
        convert_to_cam_coords: true, //TODO: check this
    };

    let vo_parameters = RuntimeParameters {
        pyramid_scale: 1.0,
        max_iterations: vec![800; 1],
        eps: vec![1e-3],
        step_sizes: vec![1e-8],
        max_norm_eps: 1e-30,
        delta_eps: 1e-30,
        taus: vec![1e-6],
        lm: true,
        weighting: true,
        debug: false,

        show_octave_result: true,
        loss_function: Box::new(loss::SoftOneLoss { eps: 1e-16, approximate_gauss_newton_matrices: true }), 
        intensity_weighting_function:  Box::new(weighting::HuberWeightForPos {delta: 1.0})
    };

    let mut se3_est = vec![Isometry3::<Float>::identity()];
    let mut se3_preintegration_est = vec![Isometry3::<Float>::identity()];

    let data_frame = d455_loader::load_data_frame(
        &root_path,
        &image_loading_parameters,
        &imu_loading_parameters,
    );
    let imu_data = &data_frame.imu_data_vec;

    se3_est.extend(run_trajectory(
        &imu_data,
        &Vector3::<Float>::new(0.0, 9.81, 0.0),
        &vo_parameters,
    ));
    let est_points = numerics::pose::apply_pose_deltas_to_point(
        Vector3::<Float>::new(0.0, 0.0, 0.0),
        &se3_est,
    );

    for (i, data) in imu_data.iter().enumerate() {
        let (imu_delta, _, _) = pre_integration(
            data,
            &Vector3::<Float>::new(0.0, 9.81, 0.0),
        );
        let pose = imu_delta.get_pose();
        se3_preintegration_est.push(pose);
    }
    let preintegration_points = numerics::pose::apply_pose_deltas_to_point(
        Vector3::<Float>::new(0.0, 0.0, 0.0),
        &se3_preintegration_est,
    );

    let out_file_name = format!("d455_imu_odom_{}_{}_{}.png", dataset_name, vo_parameters,imu_loading_parameters);

    let title = "solver";
    plot::draw_line_graph_two_vector3(
        &est_points
            .iter()
            .map(|x| Vector3::<Float>::new(x[0], x[1], x[2]))
            .collect::<Vec<Vector3<Float>>>(),
            &"estimated",
        &preintegration_points
            .iter()
            .map(|x| Vector3::<Float>::new(x[0], x[1], x[2]))
            .collect::<Vec<Vector3<Float>>>(),
            &"preintegration",
        out_folder,
        &out_file_name,
        &title,
        &"Translation",
        &"meters",
    );

    let out_preintegration_file_name = format!("d455_imu_preintegration_{}.png", dataset_name);
    let title = "preintegration";
    plot::draw_line_graph_vector3(
        &preintegration_points
            .iter()
            .map(|x| Vector3::<Float>::new(x[0], x[1], x[2]))
            .collect::<Vec<Vector3<Float>>>(),
        out_folder,
        &out_preintegration_file_name,
        &title,
        &"Translation",
        &"meters",
    );
}
