use crate::Float;
use std::fmt;

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

impl fmt::Display for DenseDirectRuntimeParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "max_iterations_{}_eps_{}_max_norm_eps_{}_delta_eps_{}_tau_{}_step_size_{}_lm_{}", self.max_iterations,self.eps,self.max_norm_eps,self.delta_eps,self.tau,self.step_size,self.lm)
    }

}