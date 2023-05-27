extern crate nalgebra as na;

use std::fmt::{Display,Debug,Formatter,Result};
use crate::Float;
use crate::numerics;


pub trait LossFunction {
    fn is_valid(&self, current_cost: Float) -> bool {
        2.0*self.second_derivative_at_current(current_cost)*current_cost+self.first_derivative_at_current(current_cost) > 0.0 + self.eps()
    }
    fn eps(&self) -> Float;
    fn root_approx(&self) -> (Float,Float) {
        let v = 1.0 - self.eps();
        (v,v)
    }
    fn cost(&self,current_cost: Float) -> Float;
    fn first_derivative_at_current(&self,current_cost: Float) -> Float;
    fn second_derivative_at_current(&self,current_cost: Float) -> Float;
    fn select_root(&self, current_cost: Float) -> Float{
        let (root_1,root_2) = match self.is_valid(current_cost) {
            true => numerics::quadratic_roots(0.5,-1.0,(-self.second_derivative_at_current(current_cost)/self.first_derivative_at_current(current_cost))*current_cost),
            false => self.root_approx()
        };
        root_1.max(root_2) //TODO: test with other root, make this customisable
    }
    fn approximate_gauss_newton_matrices(&self) -> bool;
    fn name(&self) -> &str;
}

impl Debug for dyn LossFunction + Send + Sync {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Display for dyn LossFunction + Send + Sync {

    fn fmt(&self, f: &mut Formatter) -> Result {

        let display = String::from(format!("{}_{:+e}",self.name(),self.eps()));

        write!(f, "{}", display)

    }

}


pub struct CauchyLoss {
    pub eps: Float,
    pub approximate_gauss_newton_matrices: bool
}

impl LossFunction for CauchyLoss {

    fn cost(&self, cost: Float) -> Float {
        (1.0+ cost).ln()
    }

    fn eps(&self) -> Float {
        self.eps
    }

    fn first_derivative_at_current(&self,current_cost: Float) -> Float {
        1.0/(1.0+current_cost)
    }

    fn second_derivative_at_current(&self, current_cost: Float) -> Float {
        -1.0/(1.0+current_cost).powi(2)
    }

    fn approximate_gauss_newton_matrices(&self) -> bool {
        return self.approximate_gauss_newton_matrices
    }

    fn name(&self) -> &str {
        "Cauchy"
    }

}

pub struct TrivialLoss {
    pub eps: Float,
    pub approximate_gauss_newton_matrices: bool
}

impl LossFunction for TrivialLoss {

    fn cost(&self, cost: Float) -> Float {
        cost
    }

    // roots are 0 and 2
    fn select_root(&self, _: Float) -> Float{
        0.0
    }

    fn is_valid(&self, _: Float) -> bool {
        true
    }

    fn root_approx(&self) -> (Float,Float) {
        panic!("root approx should never be called for TrivialLoss!")
    }

    fn eps(&self) -> Float {
        self.eps
    }

    fn first_derivative_at_current(&self, _: Float) -> Float {
        1.0
    }

    fn second_derivative_at_current(&self, _: Float) -> Float {
        0.0
    }

    fn approximate_gauss_newton_matrices(&self) -> bool {
        return false
    }

    fn name(&self) -> &str {
        "Trivial"
    }


}

pub struct SoftOneLoss {
    pub eps: Float,
    pub approximate_gauss_newton_matrices: bool
}


impl LossFunction for SoftOneLoss {

    fn cost(&self, cost: Float) -> Float {
        2.0*((1.0+cost).powi(2)-1.0)
    }

    fn eps(&self) -> Float {
        self.eps
    }

    fn first_derivative_at_current(&self,current_cost: Float) -> Float {
        1.0/(1.0+current_cost).sqrt()
    }

    fn second_derivative_at_current(&self, current_cost: Float) -> Float {
        -1.0/(2.0*(1.0+current_cost).powf(3.0/2.0))
    }

    fn approximate_gauss_newton_matrices(&self) -> bool {
        return self.approximate_gauss_newton_matrices
    }

    fn name(&self) -> &str {
        "SoftOneLoss"
    }

}

pub struct HuberLossForPos {
    pub eps: Float,
    pub delta: Float,
    pub approximate_gauss_newton_matrices: bool

}

// TODO: revisit this for weighting only
impl LossFunction for HuberLossForPos {

    fn cost(&self, cost: Float) -> Float {
        let cost_abs = cost.abs();
        match cost_abs {
            cost_abs if cost_abs <= self.delta => 0.5*cost_abs.powi(2),
            _ => self.delta*(cost_abs - 0.5*self.delta)
        }
    }

    fn eps(&self) -> Float {
        self.eps
    }

    fn first_derivative_at_current(&self,cost: Float) -> Float {
        let cost_abs = cost.abs();
        match cost_abs {
            cost_abs if cost_abs <= self.delta => cost,
            _ => self.delta
        }
    }

    fn second_derivative_at_current(&self, cost: Float) -> Float {
        let cost_abs = cost.abs();
        match cost_abs {
            cost_abs if cost_abs <= self.delta => 1.0,
            _ => 0.0
        }
    }

    fn approximate_gauss_newton_matrices(&self) -> bool {
        return self.approximate_gauss_newton_matrices
    }

    fn name(&self) -> &str {
        "HuberLossForPos"
    }

}


