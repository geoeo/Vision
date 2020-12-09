use crate::Float;


#[derive(Debug,Clone)]
pub struct DenseDirectRuntimeParameters{
    pub max_iterations: usize,
    pub eps: Float,
    pub max_norm_eps: Float,
    pub delta_eps: Float,
    pub tau: Float,
    pub step_size: Float,
    pub debug: bool,
    pub show_octave_result: bool,
    pub lm: bool
}