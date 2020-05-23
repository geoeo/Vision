pub mod image;
pub mod pyramid;
pub mod keypoint;

macro_rules! define_float {
    ($f:tt) => {
        use std::$f as float;
        pub type Float = $f;
    }
}

define_float!(f64);
