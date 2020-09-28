use std::fmt;

use crate::Float;

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    pub x: Float,
    pub y: Float,
    //pub sigma_level: usize#
    pub sigma: Float
} 

impl fmt::Display for ExtremaParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}, s: {}", self.x, self.y, self.sigma)
    }
}

impl ExtremaParameters {
    pub fn x_image(&self) -> usize {
        self.x.trunc() as usize
    }

    pub fn y_image(&self) -> usize {
        self.y.trunc() as usize
    }

    pub fn closest_sigma_level(&self, sigma_init: Float, s: usize) -> usize {
        //self.sigma.round() as usize
        let v = s as Float*(self.sigma/sigma_init).log2();
        //println!("{}",v);
        v.round() as usize
    }
}