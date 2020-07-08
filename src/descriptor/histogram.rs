use crate::Float;


#[derive(Debug,Clone)]
pub struct Histogram {
    pub bins: Vec<Float>
}

pub fn new(bin_size: usize) -> Histogram {
    Histogram{
        bins: vec![0.0;bin_size]
    }
}

//TODO check if this is missing a step
pub fn add_measurement(histogram: &mut Histogram, grad_orientation: (Float,Float), weight: Float) -> () {
    let grad = grad_orientation.0;
    let orientation = grad_orientation.1;
    let bin_range = 360.0/(histogram.bins.len() as Float);
    let index = (orientation/bin_range).floor() as usize;

    let weighted_grad = grad*weight; 
    histogram.bins[index] += weighted_grad;
}