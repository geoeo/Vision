use crate::Float;

#[derive(Debug,Clone)]
pub struct RGBDRuntimeParameters {
    pub sigma: Float,
    pub blur_radius: Float,
    pub octave_count: usize,
    pub use_blur: bool,
    pub min_image_dimensions: (usize,usize),
    pub invert_grad_x: bool,
    pub invert_grad_y: bool
}