extern crate nalgebra as na;

use std::fmt::{Display,Debug,Formatter,Result};
use crate::Float;


pub trait WeightingFunction {
    fn cost(&self,current_cost: Float) -> Float;
    fn name(&self) -> &str;
}

impl Debug for dyn WeightingFunction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Display for dyn WeightingFunction {

    fn fmt(&self, f: &mut Formatter) -> Result {

        let display = String::from(format!("{}",self.name()));

        write!(f, "{}", display)

    }

}



pub struct HuberWeightForPos {
    pub delta: Float
}

impl WeightingFunction for HuberWeightForPos {

    fn cost(&self, cost: Float) -> Float {
        let cost_abs = cost.abs();
        match cost_abs {
            cost_abs if cost_abs <= self.delta => 0.5*cost_abs.powi(2),
            _ => self.delta*(cost_abs - 0.5*self.delta)
        }
    }

    fn name(&self) -> &str {
        "HuberWeightForPos"
    }

}

pub struct TrivialWeight {
}

impl WeightingFunction for TrivialWeight {

    fn cost(&self, cost: Float) -> Float {
        cost
    }

    fn name(&self) -> &str {
        "TrivialWeight"
    }

}


