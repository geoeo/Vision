extern crate nalgebra as na;

use core::fmt::Debug;
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
    fn first_derivative_at_current(&self,current_cost: Float) -> Float;
    fn second_derivative_at_current(&self,current_cost: Float) -> Float;
    fn select_root(&self, current_cost: Float) -> Float{
        let (root_1,root_2) = match self.is_valid(current_cost) {
            true => numerics::quadratic_roots(0.5,-1.0,(-self.second_derivative_at_current(current_cost)/self.first_derivative_at_current(current_cost))*current_cost),
            false => self.root_approx()
        };
        root_1 //TODO: test with other root!
    }
    fn name(&self) -> &str;
}

impl Debug for dyn LossFunction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}


pub struct CauchyLoss {
    pub eps: Float
}

impl LossFunction for CauchyLoss {
    fn eps(&self) -> Float {
        self.eps
    }

    fn first_derivative_at_current(&self,current_cost: Float) -> Float {
        1.0/(1.0+current_cost)
    }

    fn second_derivative_at_current(&self, current_cost: Float) -> Float {
        -1.0/(1.0+current_cost).powi(2)
    }

    fn name(&self) -> &str {
        "CauchyLoss"
    }

}

pub struct TrivialLoss {
    pub eps: Float
}

impl LossFunction for TrivialLoss {

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

    fn first_derivative_at_current(&self,_: Float) -> Float {
        1.0
    }

    fn second_derivative_at_current(&self, _: Float) -> Float {
        0.0
    }

    fn name(&self) -> &str {
        "TrivialLoss"
    }


}



