extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use std::{fs,result::Result};

use std::path::Path;
use kiss3d::event::{WindowEvent , Key};
use kiss3d::window::Window;
use na::{Point2, Point3, Translation3};
use kiss3d::text::Font;
use vision::Float;
use vision::sfm::{bundle_adjustment::state,euclidean_landmark::EuclideanLandmark, inverse_depth_landmark::InverseLandmark};
use rand::random;

//TODO: render all states to screenshots

fn main() -> Result<(),()> {
    let mut window = Window::new("BA: Pointcloud");

    //let orb_matches_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba.txt").expect("Unable to read file");
    let final_state_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/3dv.txt").expect("Unable to read file");
    let all_states_as_string = fs::read_to_string("D:/Workspace/Rust/Vision/output/3dv_debug.txt").expect("Unable to read file");
    // let loaded: (Vec<[Float;6]>,Vec<[Float;3]>) = serde_yaml::from_str(&orb_matches_as_string).unwrap();
    // let ba_state = state::State::<EuclideanLandmark,3>::from_serial(&loaded);
    let loaded_state: (Vec<[Float;6]>,Vec<[Float;6]>) = serde_yaml::from_str(&final_state_as_string).unwrap();
    let ba_state = state::State::<InverseLandmark,6>::from_serial(&loaded_state);
    let (cams,points) = ba_state.as_matrix_point();

    let loaded_all_states: Vec<(Vec<[Float;6]>,Vec<[Float;6]>)> = serde_yaml::from_str(&all_states_as_string).unwrap();
    

    for cam in &cams {
        let cam_world = cam.inverse();
        let mut s = window.add_sphere(0.1);
        s.set_color(random(), random(), random());
        s.append_translation(&Translation3::new(cam_world.translation.vector[0] as f32,cam_world.translation.vector[1] as f32,cam_world.translation.vector[2] as f32));
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

        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(Key::P, _, _) => {
                    let image_buffer = window.snap_image();
                    let img_path = Path::new("D:/Workspace/Rust/Vision/output/screenshot.png");
                    image_buffer.save(img_path).unwrap();
                    println!("Screeshot saved to `screenshot.png`");  
                },
                _ => ()
            }
        }


    }

    Ok(())
}


