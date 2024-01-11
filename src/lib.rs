extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use serde::Deserialize;
use std::fs;
use na::{Scalar, RealField};
use simba::scalar::SubsetOf;

pub mod image;
pub mod visualize;
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

//TODO: Check SubsetOf
pub trait GenericFloat: Scalar + RealField + Copy + num_traits::float::Float + num_traits::NumAssign + num_traits::identities::One + num_traits::identities::Zero + SubsetOf<f64> {}

impl<Float: Scalar + RealField + Copy + num_traits::float::Float + num_traits::NumAssign + num_traits::identities::One + num_traits::identities::Zero + SubsetOf<f64>> GenericFloat for Float {}

#[repr(u8)]
#[derive(Debug,Copy,Clone,PartialEq)]
//TODO: Move this out of base lib.rs
pub enum GradientDirection {
    HORIZINTAL,
    VERTICAL,
    SIGMA
}

//TODO: Move this out of base lib.rs
pub fn reconstruct_original_coordiantes_for_float(x: Float, y: Float, pyramid_scaling:Float,  octave_index: i32) -> (Float,Float) {
    let factor = pyramid_scaling.powi(octave_index);
    (x*factor,y*factor)
}

//TODO: Move this out of base lib.rs
pub fn load_runtime_conf() -> RuntimeConf {
    serde_yaml::from_str(&fs::read_to_string("config.yaml").expect("No RuntimeConf found!")).expect("RuntimeConf Parsing failed!")
}

//TODO: Move this out of base lib.rs
#[derive(Deserialize, Debug)]
pub struct RuntimeConf {
    pub dataset_path: String,
    pub local_data_path: String,
    pub output_path: String
}






