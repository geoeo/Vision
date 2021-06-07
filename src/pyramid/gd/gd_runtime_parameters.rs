use crate::Float;
use std::fmt;

#[derive(Debug,Clone)]
pub struct GDRuntimeParameters {
    pub pyramid_scale: Float,
    pub sigma: Float,
    pub blur_radius: Float,
    pub octave_count: usize,
    pub use_blur: bool,
    pub min_image_dimensions: (usize,usize),
    pub invert_grad_x: bool,
    pub invert_grad_y: bool,
    pub blur_grad_x: bool,
    pub blur_grad_y: bool,
    pub normalize_gray: bool,
    pub normalize_gradients: bool
}

impl fmt::Display for GDRuntimeParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "sigma_{}_blur_radius_{}_octave_count_{}_use_blur_{}_min_image_dimensions_({},{})_invert_grad_x_{}_blur_grad_x_{}_blur_grad_x_{}_blur_grad_y_{}_normalize_gray_{}_normalize_gradients_{}", 
        self.sigma,self.blur_radius,self.octave_count,self.use_blur,self.min_image_dimensions.0,self.min_image_dimensions.1,self.invert_grad_x,self.invert_grad_y,self.blur_grad_x,self.blur_grad_y,self.normalize_gray,self.normalize_gradients)
    }

}