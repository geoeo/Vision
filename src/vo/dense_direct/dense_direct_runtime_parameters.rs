use crate::Float;


#[derive(Debug,Clone)]
pub struct DenseDirectRuntimeParameters{
    pub max_iterations: usize,
    pub eps: Float
}