extern crate nalgebra as na;
extern crate num_traits;

use std::{fmt,boxed::Box,marker::{Send,Sync}};
use crate::numerics::weighting::WeightingFunction;
use crate::GenericFloat;


#[derive(Debug)]
pub struct RuntimeParameters<F: GenericFloat>{
    pub pyramid_scale: F,
    pub max_iterations: Vec<usize>,
    pub eps: Vec<F>,
    pub max_norm_eps: F,
    pub delta_eps: F,
    pub taus: Vec<F>,
    pub step_sizes: Vec<F>,
    pub print: bool,
    pub debug: bool,
    pub show_octave_result: bool,
    pub lm: bool,
    pub intensity_weighting_function: Box<dyn WeightingFunction<F> + Send + Sync>,
    pub cg_threshold: F,
    pub cg_max_it: usize
}

impl<F: GenericFloat + fmt::LowerExp> fmt::Display for RuntimeParameters<F> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let mut display = String::from(format!("max_its_{}_w_{}",self.max_iterations[0],self.intensity_weighting_function));
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