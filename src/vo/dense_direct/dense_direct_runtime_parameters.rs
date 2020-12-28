use crate::Float;
use std::fmt;

#[derive(Debug,Clone)]
pub struct DenseDirectRuntimeParameters{
    pub max_iterations: usize,
    pub eps: Float,
    pub max_norm_eps: Float,
    pub delta_eps: Float,
    pub taus: Vec<Float>,
    pub step_sizes: Vec<Float>,
    pub debug: bool,
    pub show_octave_result: bool,
    pub lm: bool,
    pub weighting: bool
}

impl fmt::Display for DenseDirectRuntimeParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let mut display = String::from(format!("max_its_{}_w_{}",self.max_iterations,self.weighting));
        match self.lm {
            true => {
                display.push_str(format!("_lm_max_norm_eps_{:+e}_delta_eps_{:+e}",self.max_norm_eps,self.delta_eps).as_str());
                for v in &self.taus {
                    display.push_str(format!("_t_{:+e}",v).as_str());
                }
            },
            false => {
                for v in &self.step_sizes {
                    display.push_str(format!("_eps_{:+e}",self.eps).as_str());
                    display.push_str(format!("_s_s_{:+e}",v).as_str());
                }
            }

        }
        //write!(f, "max_its_{}_eps_{:+e}_max_norm_eps_{:+e}_delta_eps_{:+e}_tau_{:+e}_step_size_{}_lm_{}_w_{}", self.max_iterations,self.eps,self.max_norm_eps,self.delta_eps,self.taus,self.step_sizes,self.lm, self.weighting)
        write!(f, "{}", display)
    }

}