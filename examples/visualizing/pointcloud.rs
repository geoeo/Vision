extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use std::{fs,result::Result};

use kiss3d::window::Window;
use na::{Point2, Point3, Translation3};
use kiss3d::text::Font;
use vision::Float;
use vision::image::bundle_adjustment::state;
use vision::numerics::pose::invert_se3;
use rand::random;


fn main() -> Result<(),()> {
    let mut window = Window::new("BA: Pointcloud");

    //let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_ba_slow_1_ba_slow_2_images.txt").expect("Unable to read file");
    //let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_ba_slow_1_ba_slow_3_images.txt").expect("Unable to read file");
    let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/3dv.txt").expect("Unable to read file");
    //let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_ba_slow_1_ba_slow_2_ba_slow_3_images.txt").expect("Unable to read file");
    let loaded: (Vec<[Float;6]>,Vec<[Float;3]>) = serde_yaml::from_str(&orb_matches_as_string).unwrap();
    let ba_state = state::State::from_serial(&loaded);
    let (cams,points) = ba_state.as_matrix_point();

    


    for cam in &cams {
        let cam_world = invert_se3(&cam);
        let mut s = window.add_sphere(0.1);
        s.set_color(random(), random(), random());
        s.append_translation(&Translation3::new(cam_world[(0,3)] as f32,cam_world[(1,3)] as f32,cam_world[(2,3)] as f32));
    }

    let factor = 1.0;
    for point in &points {
        let mut s = window.add_sphere(0.02);
        s.set_color(random(), random(), random());
        s.append_translation(&Translation3::new(factor*(point[0] as f32), factor*(point[1] as f32),  factor*(point[2] as f32)));
    }


    let num_points_text = format!(
        "Number of points: {}",
        cams.len() + points.len()
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


