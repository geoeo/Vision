extern crate nalgebra as na;
extern crate vision;

use na::{Vector4,Matrix4, Vector3, UnitQuaternion};
use std::boxed::Box;
use vision::io::{loading_parameters::LoadingParameters,d455_loader};
use vision::pyramid::rgbd::{RGBDPyramid,rgbd_octave::RGBDOctave, build_rgbd_pyramid,rgbd_runtime_parameters::RGBDRuntimeParameters};
use vision::vo::{dense_direct,dense_direct::{dense_direct_runtime_parameters::DenseDirectRuntimeParameters}};
use vision::Float;
use vision::{numerics,numerics::loss};
use vision::visualize::plot;

fn main() {


    let dataset_name = "simple_trans";

    let root_path = format!("C:/Users/Marc/Workspace/Datasets/D455/{}",dataset_name);
    let out_folder = "C:/Users/Marc/Workspace/Rust/Vision/output";

    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :500,
        negate_depth_values :true,
        invert_focal_lengths :true,
        invert_y :true,
        set_default_depth: true,
        gt_alignment_rot:UnitQuaternion::identity()
    };

    let loaded_data = d455_loader::load(&root_path, &loading_parameters);

    let source_gray_images = loaded_data.source_gray_images;
    let source_depth_images = loaded_data.source_depth_images;
    let target_gray_images = loaded_data.target_gray_images;
    let target_depth_images = loaded_data.target_depth_images;
    let cam = loaded_data.pinhole_camera;

    println!("{:?}",loaded_data.pinhole_camera.projection);

    let pyramid_parameters = RGBDRuntimeParameters{
        sigma: 1.0,
        use_blur: true,
        blur_radius: 1.0,
        octave_count: 4,
        min_image_dimensions: (50,50),
        invert_grad_x : true,
        invert_grad_y : true,
        blur_grad_x : false,
        blur_grad_y: false,
        normalize_gray: true,
        normalize_gradients: false
    };

    let source_pyramids = source_gray_images.into_iter().zip(source_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<RGBDPyramid<RGBDOctave>>>();
    let target_pyramids = target_gray_images.into_iter().zip(target_depth_images.into_iter()).map(|(g,d)| build_rgbd_pyramid(g,d,&pyramid_parameters)).collect::<Vec<RGBDPyramid<RGBDOctave>>>();

    let vo_parameters = DenseDirectRuntimeParameters{
        max_iterations: vec!(500,500,500,500),
        eps: 1e-55,
        step_sizes: vec!(1.0,1.0,1.0,1.0), 
        max_norm_eps: 1e-15,
        delta_eps: 1e-15,
        taus: vec!(1e-3,1e-3,1e-3,1e-3), 
        lm: true,
        weighting: false,
        debug: true,
        show_octave_result: true,
        loss_function: Box::new(loss::CauchyLoss {eps: 1e-16})
    };

    let mut se3_est = vec!(Matrix4::<Float>::identity());
    se3_est.extend(dense_direct::run_trajectory(&source_pyramids, &target_pyramids, &cam, &vo_parameters));

    let est_points = numerics::pose::apply_pose_deltas_to_point(Vector4::<Float>::new(0.0,0.0,0.0,1.0), &se3_est);

    let out_file_name = format!("d455_{}_start_{}_counter_{}_w_{}.png",dataset_name,loading_parameters.starting_index,loading_parameters.count,vo_parameters.weighting );

    let info = "";
    plot::draw_line_graph_translation_est(&est_points.iter().map(|x| Vector3::<Float>::new(x[0],x[1], x[2])).collect::<Vec<Vector3<Float>>>(), out_folder, &out_file_name, &info);



}