use crate::Float;
use crate::features::{Feature, Oriented};

#[derive(Debug,Clone)]
pub struct KeyPoint {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize,
    pub octave_level: usize,
    pub orientation: Float
    //TODO: maybe put octave/orientation histogram here as well for debugging
}

impl Feature for KeyPoint {
    fn get_x_image(&self) -> usize {
        self.x
    }

    fn get_y_image(&self) -> usize {
        self.y
    }

    fn get_closest_sigma_level(&self) -> usize {
        self.sigma_level
    }

    fn get_id(&self) -> Option<u64> {
        None
    }
}

impl Oriented for KeyPoint {
    fn get_orientation(&self) -> Float {
        self.orientation
    }
}
