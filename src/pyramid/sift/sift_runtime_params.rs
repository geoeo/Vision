use crate::Float;

pub struct SiftRuntimeParams {
    pub pyramid_scale: Float,
    pub min_image_dimensions: (usize,usize),
    pub blur_half_factor: Float,
    pub orientation_histogram_window_factor: Float, 
    pub edge_r: Float,
    pub contrast_r: Float,
    pub sigma_initial: Float,
    pub sigma_in: Float,
    pub octave_count: usize,
    pub sigma_count: usize
}