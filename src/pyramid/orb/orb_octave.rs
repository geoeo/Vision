use crate::Float;
use crate::image::Image;

#[derive(Debug,Clone)]
pub struct OrbOctave {
    pub images: Vec<Image>,
    pub sigmas: Vec<Float>
}
