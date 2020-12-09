extern crate image as image_rs;
extern crate vision;
extern crate nalgebra as na;

use std::path::Path;
use na::Matrix4;
use vision::io::tum_loader;
use vision::pyramid::rgbd::{RGBDPyramid,rgbd_octave::RGBDOctave, build_rgbd_pyramid,rgbd_runtime_parameters::RGBDRuntimeParameters};
use vision::vo::{dense_direct,dense_direct::{dense_direct_runtime_parameters::DenseDirectRuntimeParameters}};
use vision::numerics;
use vision::Float;


fn main() {
    //let root_path = "C:/Users/Marc/Workspace/Datasets/ETH/urban_pinhole";
    let root_path = "C:/Users/Marc/Workspace/Datasets/ETH/vfr_pinhole";
    let out_folder = "output/";



    let tum_parameters = tum_loader::TUMParameters {
        starting_index: 99,
        step :1,
        count :1,
        negate_values :true,
        invert_focal_lengths :true,
        invert_y :true
    };

    let pyramid_parameters = RGBDRuntimeParameters{
    sigma: 0.1,
    use_blur: true,
    blur_radius: 1.0,
    octave_count: 1,
    min_image_dimensions: (50,50),
    invert_grad_x : true,
    blur_grad_x : false,
    invert_grad_y : true,
    blur_grad_y: false
};
    
    let tum_data = tum_loader::load(root_path, &tum_parameters);
    let source_gray_images = tum_data.source_gray_images;
    let source_depth_images = tum_data.source_depth_images;
    let target_gray_images = tum_data.target_gray_images;
    let target_depth_images = tum_data.target_depth_images;
    let cam = tum_data.pinhole_camera;

    println!("{:?}",tum_data.pinhole_camera.projection);


    let source_pyramids = source_gray_images.into_iter().zip(source_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<RGBDPyramid<RGBDOctave>>>();
    let target_pyramids = target_gray_images.into_iter().zip(target_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<RGBDPyramid<RGBDOctave>>>();


    let vo_parameters = DenseDirectRuntimeParameters{
        max_iterations: 1000,
        eps: 1e-8,
        step_size: 0.01,
        max_norm_eps: 1e-12,
        delta_eps: 1e-12,
        tau: 1e-5,
        lm: false,
        debug: true,
        show_octave_result: true
    };

    let se3_est = source_pyramids.iter().zip(target_pyramids.iter()).map(|(s,t)|  dense_direct::run(s, t,&cam , &vo_parameters)).collect::<Vec<Matrix4<Float>>>();
    let se3_gt_targetory = tum_data.source_gt_poses.iter().zip(tum_data.target_gt_poses.iter()).map(|(s,t)| {
        let se3_s = numerics::pose::se3(&s.0, &s.1);
        let se3_t = numerics::pose::se3(&t.0, &t.1);
        numerics::pose::pose_difference(&se3_s, &se3_t)
    }).collect::<Vec<Matrix4<Float>>>();

    for i in 0..se3_est.len() {
        println!("est_transform: {}",se3_est[i]);
        println!("Groundtruth Pose Delta {}",se3_gt_targetory[i]);
    }


}