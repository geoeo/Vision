use crate::Float;

pub struct RuntimeParams {
    pub blur_half_width: usize,
    pub orientation_histogram_window_size: usize, 
    pub edge_r: Float,
    pub contrast_r: Float,
    pub sigma_initial: Float,
    pub octave_count: usize,
    pub sigma_count: usize
}