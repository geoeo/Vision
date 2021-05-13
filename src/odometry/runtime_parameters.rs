use crate::Float;
use crate::numerics::loss::LossFunction;
use std::{fmt,boxed::Box};


#[derive(Debug)]
pub struct RuntimeParameters{
    pub max_iterations: Vec<usize>,
    pub eps: Vec<Float>,
    pub max_norm_eps: Float,
    pub delta_eps: Float,
    pub taus: Vec<Float>,
    pub step_sizes: Vec<Float>,
    pub debug: bool,
    pub show_octave_result: bool,
    pub lm: bool,
    pub weighting: bool,
    pub loss_function: Box<dyn LossFunction>,
    pub intensity_weighting_function: Box<dyn LossFunction>
}

impl fmt::Display for RuntimeParameters {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let mut display = String::from(format!("max_its_{}_w_{}_{}_l_{}",self.max_iterations[0],self.weighting,self.intensity_weighting_function, self.loss_function));
        match self.lm {
            true => {
                display.push_str(format!("_lm_max_norm_eps_{:+e}_delta_eps_{:+e}",self.max_norm_eps,self.delta_eps).as_str());
                for v in &self.taus {
                    display.push_str(format!("_t_{:+e}",v).as_str());
                }
            },
            false => {
                for v in &self.step_sizes {
                    display.push_str(format!("_eps_{:+e}",self.eps[0]).as_str());
                    display.push_str(format!("_s_s_{:+e}",v).as_str());
                }
            }

        }
        write!(f, "{}", display)
    }

}