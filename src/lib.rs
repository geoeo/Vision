use serde::Deserialize;

pub mod image;
pub mod visualize;
pub mod odometry;
pub mod numerics;
pub mod io;
pub mod sensors;
pub mod sfm;

macro_rules! define_float {
    ($f:tt) => {
        pub use std::$f as float;
        pub type Float = $f;
    }
}

define_float!(f64);

#[repr(u8)]
#[derive(Debug,Copy,Clone,PartialEq)]
pub enum GradientDirection {
    HORIZINTAL,
    VERTICAL,
    SIGMA
}


pub fn reconstruct_original_coordiantes_for_float(x: Float, y: Float, pyramid_scaling:Float,  octave_index: i32) -> (Float,Float) {
    let factor = pyramid_scaling.powi(octave_index);
    (x*factor,y*factor)
}

#[derive(Deserialize, Debug)]
struct Conf {
    dataset_path: String,
    local_data_path: String,
    output_path: String
}






