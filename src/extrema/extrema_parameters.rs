use std::fmt;

use crate::Float;

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    pub x: Float,
    pub y: Float,
    pub sigma_level: Float
    //pub sigma: Float
} 

impl fmt::Display for ExtremaParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}, s: {}", self.x, self.y, self.sigma_level)
    }
}

//TODO: check these
impl ExtremaParameters {
    pub fn x_image(&self) -> usize {
        self.x.trunc() as usize
    }

    pub fn y_image(&self) -> usize {
        self.y.trunc() as usize
    }

    pub fn closest_sigma_level(&self, s: usize) -> usize {
        // let sigma_range_half = (1.0/s as Float).exp2()/2.0;
        // let truncated_level =  self.sigma_level.trunc() as usize;
        // match self.sigma_level {
        //     level if level.fract() > sigma_range_half => truncated_level  + 1,
        //     _ => truncated_level
        // }
        self.sigma_level.trunc() as usize
    }
}