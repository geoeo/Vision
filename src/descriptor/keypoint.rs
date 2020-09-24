use crate::Float;

#[derive(Debug,Clone)]
pub struct KeyPoint {
    pub x: usize,
    pub y: usize,
    pub sigma_level: usize,
    pub octave_level: usize,
    pub orientation: Float
    //TODO: maybe put octave/orientation histogram here as well for debugging
} 
