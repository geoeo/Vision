extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use std::{fs,result::Result};

use kiss3d::window::Window;
use na::{Point2, Point3, Translation3};
use kiss3d::text::Font;
use vision::Float;
use vision::image::bundle_adjustment::state;
use rand::random;


fn main() -> Result<(),()> {
    let mut window = Window::new("Kiss3d: persistent_point_cloud");

    let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_ba_slow_1_ba_slow_2_images.txt").expect("Unable to read file");
    let loaded: (Vec<[Float;6]>,Vec<[Float;3]>) = serde_yaml::from_str(&orb_matches_as_string).unwrap();
    let ba_state = state::State::from_serial(&loaded);
    let (cams,points) = ba_state.lift();
    


    for cam in &cams {
        let mut s = window.add_sphere(0.01);
        s.set_color(random(), random(), random());
        s.append_translation(&Translation3::new(cam[(0,3)] as f32,cam[(1,3)] as f32,cam[(2,3)] as f32));
    }

    for point in &points {
        let mut s = window.add_sphere(0.005);
        s.set_color(random(), random(), random());
        s.append_translation(&Translation3::new(point[0] as f32,point[1] as f32, 1.5*(point[2] as f32)));
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


