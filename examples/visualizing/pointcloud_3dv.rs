extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use std::path::Path;
use std::fs::File;
use std::io::{BufReader,BufRead};
use std::{result::Result};

use kiss3d::window::Window;
use na::{Point2, Point3, Translation3, Matrix4, Vector3};
use kiss3d::text::Font;
use vision::Float;
use vision::image::bundle_adjustment::state;
use vision::numerics::pose::invert_se3;
use vision::io;
use vision::numerics::lie;
use vision::numerics::pose;
use rand::random;


fn main() -> Result<(),()> {
    let mut window = Window::new("BA: Pointcloud");



    let file_path_str = format!("D:/Workspace/Cpp/3dv_tutorial/bin/bundle_adjustment_global(point)_neg_z_lm.xyz");
    let file_cam_path_str = format!("D:/Workspace/Cpp/3dv_tutorial/bin/bundle_adjustment_global(camera)_neg_z_lm.xyz");
    let file_path = Path::new(&file_path_str);
    let file_cam_path = Path::new(&file_cam_path_str);
    let file = File::open(file_path).expect(format!("Could not open: {}", file_path.display()).as_str());
    let file_cam = File::open(file_cam_path).expect(format!("Could not open: {}", file_path.display()).as_str());



    let reader = BufReader::new(file);
    let lines = reader.lines();
    let points = lines.map(|l| {
        let v = l.unwrap();
        let values = v.trim().split(' ').collect::<Vec<&str>>();
        let x = io::parse_to_float(values[0], false); 
        let y = io::parse_to_float(values[1], false); 
        let z = io::parse_to_float(values[2], false); 
        Point3::<Float>::new(x,y,z)
    }).collect::<Vec<Point3<Float>>>();

    let reader_cam = BufReader::new(file_cam);
    let lines_cam = reader_cam.lines();
    let cams = lines_cam.map(|l| {
        let v = l.unwrap();
        let values = v.trim().split(' ').collect::<Vec<&str>>();
        let w_1 = io::parse_to_float(values[0], false); 
        let w_2 = io::parse_to_float(values[1], false); 
        let w_3 = io::parse_to_float(values[2], false);
        let u_1 = io::parse_to_float(values[3], false); 
        let u_2 = io::parse_to_float(values[4], false); 
        let u_3 = io::parse_to_float(values[5], false); 
        let w = Vector3::new(w_1,w_2,w_3);
        let u = Vector3::new(u_1,u_2,u_3);
        let cam = lie::exp(&u, &w);
        pose::invert_se3(&cam)


    }).collect::<Vec<Matrix4<Float>>>();


    for cam_world in cams {
        let mut s = window.add_sphere(0.01);
        s.set_color(random(), random(), random());
        s.append_translation(&Translation3::new(cam_world[(0,3)] as f32,cam_world[(1,3)] as f32,cam_world[(2,3)] as f32));
    }



    let factor = 10.0;
    for point in &points {
        let mut s = window.add_sphere(0.01);
        s.set_color(random(), random(), random());
        s.append_translation(&Translation3::new(factor*(point[0] as f32), factor*(point[1] as f32),factor - 1.0 + factor/5.0*(point[2] as f32)));
    }


    let num_points_text = format!(
        "Number of points: {}",
        points.len()
    );
    while window.render() {
        window.draw_text(
            &num_points_text,
            &Point2::new(0.0, 20.0),
            60.0,
            &Font::default(),
            &Point3::new(1.0, 1.0, 1.0),
        );
    }

    Ok(())
}


