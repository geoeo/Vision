extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;

use na::{Vector4,Matrix4, Vector3, UnitQuaternion};
use vision::io::{loading_parameters::LoadingParameters,tum_loader};
use vision::pyramid::rgbd::{RGBDPyramid,rgbd_octave::RGBDOctave, build_rgbd_pyramid,rgbd_runtime_parameters::RGBDRuntimeParameters};
use vision::vo::{dense_direct,dense_direct::{dense_direct_runtime_parameters::DenseDirectRuntimeParameters}};
use vision::numerics;
use vision::{Float,float};
use vision::visualize::plot;


fn main() {
    let root_path = "C:/Users/Marc/Workspace/Datasets/TUM/rgbd_dataset_freiburg2_desk";
    let dataset = tum_loader::Dataset::FR2;
    let out_folder = "output/";


    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :300,
        negate_values :true,
        invert_focal_lengths :true,
        invert_y :true,
        gt_alignment: UnitQuaternion::<Float>::from_axis_angle(&Vector3::y_axis(),float::consts::PI)
    };



    let pyramid_parameters = RGBDRuntimeParameters{
    sigma: 0.1,
    use_blur: true,
    blur_radius: 1.0,
    octave_count: 3,
    min_image_dimensions: (50,50),
    invert_grad_x : true,
    invert_grad_y : true,
    blur_grad_x : true,
    blur_grad_y: true,
    normalize_gray: true,
    normalize_gradients: false
};
    
    let tum_data = tum_loader::load(root_path, &loading_parameters,&dataset);
    let source_gray_images = tum_data.source_gray_images;
    let source_depth_images = tum_data.source_depth_images;
    let target_gray_images = tum_data.target_gray_images;
    let target_depth_images = tum_data.target_depth_images;
    let cam = tum_data.pinhole_camera;

    println!("{:?}",tum_data.pinhole_camera.projection);


    let source_pyramids = source_gray_images.into_iter().zip(source_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<RGBDPyramid<RGBDOctave>>>();
    let target_pyramids = target_gray_images.into_iter().zip(target_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<RGBDPyramid<RGBDOctave>>>();


    let vo_parameters = DenseDirectRuntimeParameters{
        max_iterations: 500,
        eps: 1e-7,
        step_size: 1.0, //TODO make these paramters per octave level
        max_norm_eps: 1e-25,
        delta_eps: 1e-25,
        tau: 1e-3,
        lm: true,
        debug: true,
        show_octave_result: true
    };

    let mut se3_est = vec!(Matrix4::<Float>::identity());
    let mut se3_gt_targetory = vec!(Matrix4::<Float>::identity());



    se3_est.extend(source_pyramids.iter().zip(target_pyramids.iter()).map(|(s,t)|  dense_direct::run(s, t,&cam , &vo_parameters)).collect::<Vec<Matrix4<Float>>>());
    se3_gt_targetory.extend(tum_data.source_gt_poses.iter().zip(tum_data.target_gt_poses.iter()).map(|(s,t)| {
        let se3_s = numerics::pose::se3(&s.0, &s.1);
        let se3_t = numerics::pose::se3(&t.0, &t.1);
        numerics::pose::pose_difference(&se3_s, &se3_t)
    }).collect::<Vec<Matrix4<Float>>>());

    let est_points = numerics::pose::apply_pose_deltas_to_point(Vector4::<Float>::new(0.0,0.0,0.0,1.0), &se3_est);
    let est_gt_points = numerics::pose::apply_pose_deltas_to_point(Vector4::<Float>::new(0.0,0.0,0.0,1.0), &se3_gt_targetory);


    // for i in 0..se3_est.len() {
    //     println!("est_transform: {}",se3_est[i]);
    //     println!("Groundtruth Pose Delta {}",se3_gt_targetory[i]);
    // }



    // for i in 0..est_points.len() {
    //     println!("Point trajectory: {}",est_points[i]);
    //     println!("Gt Trajectory {}",est_gt_points[i]);
    // }

    //plot::draw_line_graph_est_gt(&est_points.iter().map(|point| point[0]).collect::<Vec<Float>>(),&est_gt_points.iter().map(|point| point[0]).collect::<Vec<Float>>(), out_folder, "x-translation.png");
    plot::draw_line_graph_translation_est_gt(&est_points.iter().map(|x| Vector3::<Float>::new(x[0],x[1], x[2])).collect::<Vec<Vector3<Float>>>(),&est_gt_points.iter().map(|x| Vector3::<Float>::new(x[0],x[1], x[2])).collect::<Vec<Vector3<Float>>>(), out_folder, "translation.png");



}