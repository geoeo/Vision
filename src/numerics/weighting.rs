extern crate nalgebra as na;

use na::DVector;
use std::fmt::{Display,Debug,Formatter,Result};
use crate::{float,Float};
use crate::numerics::estimate_std;


pub trait WeightingFunction {
    fn weight(&self, residuals: &DVector<Float>, index: usize, std: Option<Float>) -> Float;
    fn estimate_standard_deviation(&self, residuals: &DVector<Float>) -> Option<Float>;
    fn cost(&self, residuals: &DVector<Float>, std: Option<Float>) -> Float;
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

//TODO: revisit these and include references for M-estimators
pub struct HuberWeightForPos {
}

impl WeightingFunction for HuberWeightForPos {

    fn weight(&self, residuals: &DVector<Float>, index: usize,  std: Option<Float>) -> Float {
        let res_abs = residuals[index].abs();
        let k = std.expect("k has to have been computed for Huber Weight");
        match res_abs {
            v if (v <= k || k == 0.0) => 1.0,
            _ => k/res_abs
        }
    }

    fn estimate_standard_deviation(&self, residuals: &DVector<Float>) -> Option<Float> {
        Some(1.345*estimate_std(residuals))
    }

    fn name(&self) -> &str {
        "HuberWeightForPos"
    }

    fn cost(&self, residuals: &DVector<Float>, std: Option<Float>) -> Float {
        let k = std.expect("k has to have been computed for Huber Weight");
        residuals.map(|e| {
            let e_abs = e.abs();
            match e_abs {
                v if v <= k => e.powi(2)/2.0,
                v if v > k => k*(v - (e/2.0)),
                _ => 0.0
            }
        }).sum()
    }

}

pub struct CauchyWeight {
    pub c: Float
}

impl WeightingFunction for CauchyWeight {

    fn weight(&self, residuals: &DVector<Float>, index: usize,  _ : Option<Float>) -> Float {
        let res = residuals[index];
        1.0/(1.0+(res/self.c).powi(2))
    }

    fn estimate_standard_deviation(&self, _: &DVector<Float>) -> Option<Float> {
        None
    }

    fn name(&self) -> &str {
        "CauchyWeight"
    }

    fn cost(&self, residuals: &DVector<Float>, _: Option<Float>) -> Float {
        let factor = self.c.powi(2)/2.0;
        factor*residuals.map(|e| (1.0+ e.powi(2)/self.c).ln()).sum()
    }

}


pub struct SquaredWeight {
}

impl WeightingFunction for SquaredWeight {

    fn weight(&self,_residuals: &DVector<Float>, _index: usize,  _: Option<Float>) -> Float {
        1.0
    }

    fn estimate_standard_deviation(&self, _residuals: &DVector<Float>) -> Option<Float> {
        None
    }

    fn name(&self) -> &str {
        "TrivialWeight"
    }

    fn cost(&self, residuals: &DVector<Float>, _: Option<Float>) -> Float {
        (residuals.transpose() * residuals)[0]
    }

}

pub struct TDistWeight {
    pub t_dist_nu: Float,
    pub max_it: usize,
    pub eps: Float
}

//From Engels Paper -> Check cost
impl TDistWeight {
    fn estimate_t_dist_variance(&self, residuals: &DVector<Float>, t_dist_nu: Float, max_it: usize, eps: Float) -> Float {
        let mut it = 0;
        let mut err = float::MAX;
        let mut variance = float::MAX; 
        let mut n = 0.0;
    
        while it < max_it && err > eps {
            let mut acc = 0.0;
            for r in residuals {
                if *r == 0.0 {
                    continue;
                }
                let r_sqrd = r.powi(2);
                acc += r_sqrd*(t_dist_nu +1.0)/(t_dist_nu + r_sqrd/variance);
                n+=1.0;
            }
    
            let var_old = variance;
            variance = acc/n;
            err = (variance-var_old).abs();
            it += 1;
        }
    
        variance
    }

    fn compute_t_dist_weight(&self, residual: Float, variance: Float, t_dist_nu: Float) -> Float {
        (t_dist_nu + 1.0) / (t_dist_nu + residual.powi(2)/variance)
    }
}

impl WeightingFunction for TDistWeight {
    fn weight(&self, residuals: &DVector<Float>, index: usize,  variance: Option<Float>) -> Float {
        let variance = variance.expect("Variance should have been computed for T-Dist weighting!");
        self.compute_t_dist_weight(residuals[index],variance, self.t_dist_nu).sqrt()
    }

    fn estimate_standard_deviation(&self, residuals: &DVector<Float>) -> Option<Float> {
        Some(self.estimate_t_dist_variance(residuals, self.t_dist_nu,self.max_it,self.eps))
    }

    fn name(&self) -> &str {
        "T-Dist"
    }

    fn cost(&self, residuals: &DVector<Float>, _: Option<Float>) -> Float {
        (residuals.transpose() * residuals)[0]
    }

}


