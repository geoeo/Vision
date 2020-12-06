use crate::Float;


#[derive(Debug,Clone)]
pub struct DenseDirectRuntimeParameters{
    pub max_iterations: usize,
    pub eps: Float,
    pub initial_step_size: Float,
    pub invert_y: bool,
    pub invert_grad: bool
}