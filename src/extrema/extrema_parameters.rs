use std::fmt;

#[derive(Debug,Clone)]
pub struct ExtremaParameters {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize
} 

impl fmt::Display for ExtremaParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}, s: {}", self.x, self.y, self.sigma_level)
    }
}