extern crate nalgebra as na;
extern crate num_traits;

use std::fmt::{Display,Debug,Formatter,Result};
use crate::numerics::estimate_std;
use num_traits::{float,NumAssign, pow, identities};
use na::{convert, SimdRealField, DVector,base::Scalar};


pub trait WeightingFunction<F : float::Float + Scalar + NumAssign + SimdRealField > {
    fn weight(&self, residuals: &DVector<F>, index: usize, std: Option<F>) -> F;
    fn estimate_standard_deviation(&self, residuals: &DVector<F>) -> Option<F>;
    fn cost(&self, residuals: &DVector<F>) -> F;
    fn name(&self) -> &str;
}

impl<F> Debug for dyn WeightingFunction<F> where F : float::Float + Scalar + NumAssign + SimdRealField {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<F> Display for dyn WeightingFunction<F> where F : float::Float + Scalar + NumAssign + SimdRealField {

    fn fmt(&self, f: &mut Formatter) -> Result {

        let display = String::from(format!("{}",self.name()));

        write!(f, "{}", display)

    }

}

//TODO: revisit these and include references for M-estimators
pub struct HuberWeight {
}

impl<F> WeightingFunction<F> for HuberWeight where F : float::Float + Scalar + NumAssign + SimdRealField + identities::One{

    fn weight(&self, residuals: &DVector<F>, index: usize,  std: Option<F>) -> F {
        let res_abs = float::Float::abs(residuals[index]);
        let k = std.expect("k has to have been computed for Huber Weight");
        match res_abs {
            v if v <= k => F::one(),
            _ => k/res_abs
        }
    }

    fn estimate_standard_deviation(&self, residuals: &DVector<F>) -> Option<F> {
        Some(convert::<f64,F>(1.345)*estimate_std(residuals))
    }

    fn name(&self) -> &str {
        "HuberWeight"
    }

    fn cost(&self, residuals: &DVector<F>) -> F {
        float::Float::sqrt((residuals.transpose() * residuals)[0])
    }

}

pub struct CauchyWeight<F> where F : float::Float + Scalar + NumAssign + SimdRealField + identities::One {
    pub c: F
}

impl<F> WeightingFunction<F> for CauchyWeight<F> where F : float::Float + Scalar + NumAssign + SimdRealField + identities::One{

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

    fn cost(&self, residuals: &DVector<F>) -> F {
        float::Float::sqrt((residuals.transpose() * residuals)[0])
    }

}

pub struct BisquareWeight {
    
}

//TODO: investigate this
impl<F> WeightingFunction<F> for BisquareWeight where F : float::Float + Scalar + NumAssign + SimdRealField + identities::One {

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

    fn cost(&self, residuals: &DVector<F>) -> F {
        float::Float::sqrt((residuals.transpose() * residuals)[0])
    }

}



pub struct SquaredWeight {
}

impl<F> WeightingFunction<F> for SquaredWeight where F : float::Float + Scalar + NumAssign + SimdRealField + identities::One {

    fn weight(&self,_residuals: &DVector<F>, _index: usize,  _: Option<F>) -> F {
        F::one()
    }

    fn estimate_standard_deviation(&self, _residuals: &DVector<F>) -> Option<F> {
        None
    }

    fn name(&self) -> &str {
        "TrivialWeight"
    }

    fn cost(&self, residuals: &DVector<F>) -> F {
        convert::<f64,F>(0.5)*(residuals.transpose() * residuals)[0]
    }

}

// pub struct TDistWeight {
//     pub t_dist_nu: Float,
//     pub max_it: usize,
//     pub eps: Float
// }

//From Engels Paper -> Check cost
// impl TDistWeight {
//     fn estimate_t_dist_variance(&self, residuals: &DVector<Float>, t_dist_nu: Float, max_it: usize, eps: Float) -> Float {
//         let mut it = 0;
//         let mut err = crate::float::MAX;
//         let mut variance = crate::float::MAX; 
//         let mut n = 0.0;
    
//         while it < max_it && err > eps {
//             let mut acc = 0.0;
//             for r in residuals {
//                 if *r == 0.0 {
//                     continue;
//                 }
//                 let r_sqrd = r.powi(2);
//                 acc += r_sqrd*(t_dist_nu +1.0)/(t_dist_nu + r_sqrd/variance);
//                 n+=1.0;
//             }
    
//             let var_old = variance;
//             variance = acc/n;
//             err = (variance-var_old).abs();
//             it += 1;
//         }
    
//         variance
//     }

//     fn compute_t_dist_weight(&self, residual: F, variance: F, t_dist_nu: F) -> F {
//         (t_dist_nu + 1.0) / (t_dist_nu + residual.powi(2)/variance)
//     }
// }

// impl<F> WeightingFunction<F> for TDistWeight where F : float::Float + Scalar + NumAssign + SimdRealField + ComplexField {
//     fn weight(&self, residuals: &DVector<F>, index: usize,  variance: Option<F>) -> F {
//         let variance = variance.expect("Variance should have been computed for T-Dist weighting!");
//         self.compute_t_dist_weight(residuals[index],variance, self.t_dist_nu).sqrt()
//     }

//     fn estimate_standard_deviation(&self, residuals: &DVector<F>) -> Option<F> {
//         Some(self.estimate_t_dist_variance(residuals, self.t_dist_nu,self.max_it,self.eps))
//     }

//     fn name(&self) -> &str {
//         "T-Dist"
//     }

//     fn cost(&self, residuals: &DVector<F>) -> F {
//         ((residuals.transpose() * residuals)[0]).sqrt()
//     }

// }


