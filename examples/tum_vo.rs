extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;

use na::{Vector4,Matrix4, Vector3, UnitQuaternion};
use std::boxed::Box;
use vision::io::{image_loading_parameters::ImageLoadingParameters,tum_loader};
use vision::pyramid::gd::{GDPyramid,gd_octave::GDOctave, build_rgbd_pyramid,gd_runtime_parameters::GDRuntimeParameters};
use vision::odometry::visual_odometry::dense_direct;
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics;
use vision::{Float,float};
use vision::visualize::plot;



fn main() {
    //let dataset_name = "freiburg3_long_office_household";
    //let dataset_name = "freiburg3_structure_texture_near";
    let dataset_name = "freiburg2_desk";
    //let dataset_name = "freiburg2_rpy";

    let root_path = format!("C:/Users/Marc/Workspace/Datasets/TUM/rgbd_dataset_{}",dataset_name);
    let dataset = tum_loader::Dataset::FR2;
    let out_folder = "C:/Users/Marc/Workspace/Rust/Vision/output";


    let loading_parameters = ImageLoadingParameters {
        starting_index: 0,
        step :1,
        count :150,
        negate_depth_values :false,
        invert_focal_lengths :false,
        invert_y :true,
        set_default_depth: true,
        gt_alignment_rot:UnitQuaternion::<Float>::from_axis_angle(&Vector3::x_axis(),float::consts::FRAC_PI_2)* UnitQuaternion::<Float>::from_axis_angle(&Vector3::y_axis(),float::consts::PI)
    };



    let pyramid_parameters = GDRuntimeParameters{
        sigma: 0.01,
        use_blur: true,
        blur_radius: 1.0,
        octave_count: 4,
        min_image_dimensions: (50,50),
        invert_grad_x : true,
        invert_grad_y : true,
        blur_grad_x : false, //TODO: make bluring gradient cleaner
        blur_grad_y: false,
        normalize_gray: true,
        normalize_gradients: false
    };
    
    let tum_data = tum_loader::load(&root_path, &loading_parameters,&dataset);
    let source_gray_images = tum_data.source_gray_images;
    let source_depth_images = tum_data.source_depth_images;
    let target_gray_images = tum_data.target_gray_images;
    let target_depth_images = tum_data.target_depth_images;
    let intensity_cam = tum_data.intensity_camera;
    let depth_cam = tum_data.intensity_camera;

    println!("{:?}",tum_data.intensity_camera.projection);


    let source_pyramids = source_gray_images.into_iter().zip(source_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<GDPyramid<GDOctave>>>();
    let target_pyramids = target_gray_images.into_iter().zip(target_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<GDPyramid<GDOctave>>>();


    let vo_parameters = RuntimeParameters{
        max_iterations: vec![800;4],
        eps: vec!(1e-5,9e-4,9e-4,1e-6),
        step_sizes: vec!(1e-0,5e-1,5e-1,1e-3), 
        max_norm_eps: 1e-10,
        delta_eps: 1e-10,
        taus: vec!(1e-6,1e-3,1e-3,1e-0), 
        lm: true,
        weighting: true,
        debug: false,

        show_octave_result: true,
        loss_function: Box::new(numerics::loss::CauchyLoss {eps: 1e-16})
    };

    let mut se3_est = vec!(Matrix4::<Float>::identity());
    let mut se3_gt_targetory = vec!(Matrix4::<Float>::identity());


    se3_est.extend(dense_direct::solver::run_trajectory(&source_pyramids, &target_pyramids, &intensity_cam, &depth_cam, &vo_parameters));
    se3_gt_targetory.extend(tum_data.source_gt_poses.unwrap().iter().zip(tum_data.target_gt_poses.unwrap().iter()).map(|(s,t)| {
        let se3_s = numerics::pose::se3(&s.0, &s.1);
        let se3_t = numerics::pose::se3(&t.0, &t.1);
        numerics::pose::pose_difference(&se3_s, &se3_t)
    }).collect::<Vec<Matrix4<Float>>>());

    let est_points = numerics::pose::apply_pose_deltas_to_point(Vector4::<Float>::new(0.0,0.0,0.0,1.0), &se3_est);
    let est_gt_points = numerics::pose::apply_pose_deltas_to_point(Vector4::<Float>::new(0.0,0.0,0.0,1.0), &se3_gt_targetory);
    let mut errors = Vec::<Matrix4<Float>>::with_capacity(se3_est.len()-1);
    for i in 0..se3_est.len()-loading_parameters.step{
        let p_1 = se3_est[i];
        let p_2 = se3_est[i+loading_parameters.step];
        let q_1 = se3_gt_targetory[i];
        let q_2 = se3_gt_targetory[i+loading_parameters.step];

        errors.push(numerics::pose::error(&q_1,&q_2,&p_1,&p_2));
    }

    let rmse = numerics::pose::rsme(&errors);

    let out_file_name = format!("{}_{}_{}_s_{}_o_{}_b_{}_br_{}_neg_d_{}.png",dataset_name,loading_parameters.starting_index,vo_parameters,pyramid_parameters.sigma,pyramid_parameters.octave_count, pyramid_parameters.use_blur,pyramid_parameters.blur_radius, loading_parameters.negate_depth_values);
    //let info = format!("{}_{}_{}",loading_parameters,pyramid_parameters,vo_parameters);
    let info = format!("rsme: {}",rmse);
    plot::draw_line_graph_translation_est_gt(&est_points.iter().map(|x| Vector3::<Float>::new(x[0],x[1], x[2])).collect::<Vec<Vector3<Float>>>(),&est_gt_points.iter().map(|x| Vector3::<Float>::new(x[0],x[1], x[2])).collect::<Vec<Vector3<Float>>>(), out_folder, &out_file_name, &info);



}