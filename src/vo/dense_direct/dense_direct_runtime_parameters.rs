use crate::Float;


#[derive(Debug,Clone)]
pub struct DenseDirectRuntimeParameters{
    pub max_iterations: usize,
    pub eps: Float,
    pub max_norm_eps: Float,
    pub delta_eps: Float,
    pub tau: Float,
    pub initial_step_size: Float,
    pub invert_y: bool,
    pub invert_grad: bool,
    pub debug: bool
}