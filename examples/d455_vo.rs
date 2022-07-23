extern crate nalgebra as na;
extern crate vision;

use na::{Point3, UnitQuaternion, Isometry3};
use std::boxed::Box;
use vision::io::{image_loading_parameters::ImageLoadingParameters,d455_loader};
use vision::image::pyramid::gd::{GDPyramid,gd_octave::GDOctave, build_rgbd_pyramid,gd_runtime_parameters::GDRuntimeParameters};
use vision::odometry::visual_odometry::dense_direct;
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::Float;
use vision::{numerics, numerics::{loss,weighting}};
use vision::visualize::plot;

fn main() {


    let dataset_name = "x";

    let root_path = format!("D:/Workspace/Datasets/D455/{}",dataset_name);
    let out_folder = "D:/Workspace/Rust/Vision/output";

    let loading_parameters = ImageLoadingParameters {
        starting_index: 5,
        step: 1,
        count: 70,
        image_height: 480,
        image_width: 640,
        negate_depth_values :true,
        invert_focal_lengths :true,
        invert_y :true,
        set_default_depth: true,
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
        pyramid_scale: 1.2,
        sigma: 0.1,
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

    let source_pyramids = source_gray_images.into_iter().zip(source_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<GDPyramid<GDOctave>>>();
    let target_pyramids = target_gray_images.into_iter().zip(target_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<GDPyramid<GDOctave>>>();

    let vo_parameters = RuntimeParameters{
        pyramid_scale: pyramid_parameters.pyramid_scale,
        max_iterations: vec![50;3],
        eps: vec!(1e-3,1e-3,1e-3),
        step_sizes: vec!(1e-8,1e-8,1e-8), 
        max_norm_eps: 1e-10,
        delta_eps: 1e-10,
        taus: vec!(1e-6,1e-3,1e0), 
        lm: true,
        debug: false,

        show_octave_result: true,
        loss_function: Box::new(loss::TrivialLoss {eps: 1e-16, approximate_gauss_newton_matrices: true}),
        intensity_weighting_function:  Box::new(weighting::HuberWeight {}),
        cg_threshold: 1e-6,
        cg_max_it: 200
    };

    let mut se3_est = vec!(Isometry3::<Float>::identity());
    se3_est.extend(dense_direct::solver::run_trajectory(&source_pyramids, &target_pyramids, &intensity_cam, &depth_cam, &vo_parameters));

    let est_points = numerics::pose::apply_pose_deltas_to_point(Point3::<Float>::new(0.0,0.0,0.0), &se3_est);

    let out_file_name = format!("d455_vo_{}_start_{}_counter_{}_{}.png",dataset_name,loading_parameters.starting_index,loading_parameters.count,vo_parameters);

    let title = "d455";
    plot::draw_line_graph_vector3(&est_points, out_folder, &out_file_name, &title, &"Translation", &"meters");



}