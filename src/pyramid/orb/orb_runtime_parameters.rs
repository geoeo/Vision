use crate::Float;

pub struct OrbRuntimeParameters {
    pub sigma: Float,
    pub blur_radius: Float,
    pub octave_count: usize,
    pub max_features_per_octave: usize,
    pub min_image_dimensions: (usize,usize),
    pub harris_k: Float,
    pub fast_circle_radius: usize,
    pub fast_threshold_factor: Float,
    pub fast_consecutive_pixels: usize,
    pub fast_grid_size: (usize,usize),
    pub fast_grid_size_scale_base: Float,
    pub fast_offsets: (usize,usize),
    pub fast_offset_scale_base: Float,
    pub brief_features_to_descriptors: usize,
    pub brief_n: usize,
    pub brief_s: usize,
    pub brief_s_scale_base: Float,
    pub brief_matching_min_threshold: u64,
    pub brief_lookup_table_step: Float,
}