extern crate nalgebra as na;
extern crate num_traits;

use std::fmt::{Display,Debug,Formatter,Result};
use crate::numerics::estimate_std;
use num_traits::{float, pow};
use na::{convert, DVector};
use crate::GenericFloat;


pub trait WeightingFunction<F : GenericFloat > {
    fn weight(&self, residuals: &DVector<F>, index: usize, std: Option<F>) -> F;
    fn estimate_standard_deviation(&self, residuals: &DVector<F>) -> Option<F>;
    fn cost(&self, residuals: &DVector<F>, std: Option<F>) -> F;
    fn name(&self) -> &str;
}

impl<F> Debug for dyn WeightingFunction<F>  + Send + Sync where F : GenericFloat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<F> Display for dyn WeightingFunction<F>  + Send + Sync where F : GenericFloat {

    fn fmt(&self, f: &mut Formatter) -> Result {

        let display = String::from(format!("{}",self.name()));

        write!(f, "{}", display)

    }

}

//TODO: revisit these and include references for M-estimators
pub struct HuberWeight {
}

impl<F> WeightingFunction<F> for HuberWeight where F : GenericFloat {

    fn weight(&self, residuals: &DVector<F>, index: usize,  std: Option<F>) -> F {
        let res_abs = float::Float::abs(residuals[index]);
        let k = convert::<f64,F>(1.345)*std.expect("k has to have been computed for Huber Weight");
        match res_abs {
            v if v <= k => F::one(),
            _ => k/res_abs
        }
    }

    fn estimate_standard_deviation(&self, residuals: &DVector<F>) -> Option<F> {
        Some(convert::<f64,F>(1.345)*(estimate_std(residuals) + convert::<f64,F>(1e3))) // small delta so we dont return 0
    }

    fn name(&self) -> &str {
        "HuberWeight"
    }

    fn cost(&self, residuals: &DVector<F>, std: Option<F>) -> F {
        let half = convert::<f64,F>(0.5);
        let k = convert::<f64,F>(1.345)*std.expect("k has to have been computed for Huber Weight");
        residuals.iter().map(|&e| {
            match float::Float::abs(e) {
                e_abs if e_abs <= k => half* float::Float::powi(e,2),
                _ => k*float::Float::abs(e)-half*float::Float::powi(k,2)
            }
        }).sum()
    }

}

//Incorrect
pub struct CauchyWeight<F> where F : GenericFloat {
    pub c: F
}

impl<F> WeightingFunction<F> for CauchyWeight<F> where F : GenericFloat {

    fn weight(&self, residuals: &DVector<F>, index: usize,  _ : Option<F>) -> F {
        let res = residuals[index];
        pow::pow(F::one()/(F::one()+(res/self.c)),2)
    }

    fn estimate_standard_deviation(&self, _: &DVector<F>) -> Option<F> {
        None
    }

    fn name(&self) -> &str {
        "CauchyWeight"
    }

    fn cost(&self, residuals: &DVector<F>, _ :Option<F>) -> F {
        float::Float::sqrt((residuals.transpose() * residuals)[0])
    }

}

//Incorrect
pub struct BisquareWeight {
    
}

//TODO: investigate this
impl<F> WeightingFunction<F> for BisquareWeight where F : GenericFloat {

    fn weight(&self, residuals: &DVector<F>, index: usize,  std: Option<F>) -> F {
        let e = residuals[index];
        let e_abs = float::Float::abs(e);
        let k = std.expect("k has to have been computed for Huber Weight");
        match e_abs {

            v if v <= k => {
                let t = float::Float::powi(e/k,2);
                F::one() - float::Float::powi(t,2)
            },
            _ => F::zero()
        }
    }

    fn estimate_standard_deviation(&self, residuals: &DVector<F>) -> Option<F> {
        Some(convert::<f64,F>(4.685)*estimate_std(residuals))
    }

    fn name(&self) -> &str {
        "Bisquare"
    }

    fn cost(&self, residuals: &DVector<F>, _:  Option<F>) -> F {
        float::Float::sqrt((residuals.transpose() * residuals)[0])
    }

}



pub struct SquaredWeight {
}

impl<F> WeightingFunction<F> for SquaredWeight where F : GenericFloat {

    fn weight(&self,_residuals: &DVector<F>, _index: usize,  _: Option<F>) -> F {
        F::one()
    }

    fn estimate_standard_deviation(&self, _residuals: &DVector<F>) -> Option<F> {
        None
    }

    fn name(&self) -> &str {
        "SquaredWeight"
    }

    fn cost(&self, residuals: &DVector<F>, _: Option<F>) -> F {
        convert::<f64,F>(0.5)*(residuals.transpose() * residuals)[0]
    }

}


