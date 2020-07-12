pub mod image;
pub mod pyramid;
pub mod keypoint;
pub mod descriptor;

macro_rules! define_float {
    ($f:tt) => {
        use std::$f as float;
        pub type Float = $f;
    }
}

define_float!(f64);

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize
} 

#[derive(Debug,Clone)]
pub struct KeyPoint {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize,
    pub orientation: Float
} 

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum GradientDirection {
    HORIZINTAL,
    VERTICAL,
    SIGMA
}
