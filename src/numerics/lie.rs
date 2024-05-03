extern crate nalgebra as na;

use num_traits::Float;
use na::{Vector,Vector3,Vector6,Matrix3,Matrix3x6,Matrix, Matrix4,U3,U1,base::storage::Storage, convert};
use crate::GenericFloat;

pub fn vector_from_skew_symmetric<F>(w_x: &Matrix3<F>) -> Vector3<F> where F : GenericFloat {
    Vector3::<F>::new(w_x[(2,1)],w_x[(0,2)],w_x[(1,0)])
}

/**
 * This is the jacobian with respect to a transformed point
 */
pub fn left_jacobian_around_identity<F, T>(transformed_position: &Vector<F,U3,T>) -> Matrix3x6<F> 
    where 
    T: Storage<F,U3,U1>, 
    F : GenericFloat {
    let skew_symmetrix = &(-transformed_position).cross_matrix();
    let mut jacobian = Matrix3x6::<F>::new(
        F::one(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero(),
        F::zero(), F::one(), F::zero(), F::zero(), F::zero(), F::zero(),
        F::zero(), F::zero(), F::one(), F::zero(), F::zero(), F::zero());
    
    for i in 3..6 {
        jacobian.set_column(i, &skew_symmetrix.column(i-3));
    }

    jacobian

}

//Improvement: taylor expansion
#[allow(non_snake_case)]
pub fn exp_se3<F, T>(u: &Vector<F,U3,T>, w: &Vector<F,U3,T>) -> Matrix4<F> where T: Storage<F,U3,U1>, F : GenericFloat {
    let omega = Float::sqrt((w.transpose()*w)[0]);
    let omega_sqr = Float::powi(omega,2);

    let (A,B,C) = match omega {
        omega if omega != F::zero() => {
            let A = Float::sin(omega)/omega;
            (A,(F::one() - Float::cos(omega))/omega_sqr, (F::one() - A)/omega_sqr)
        },
        _ => (F::one(),F::one(),F::one())
    };

    let w_x = w.cross_matrix();
    let w_x_sqr = w_x*w_x;
    let I = Matrix3::<F>::identity();
    let R = I + w_x.scale(A) + w_x_sqr.scale(B);
    let V = I + w_x.scale(B) + w_x_sqr.scale(C);
    let t = V*u;

    let mut res = Matrix4::<F>::identity();
    let mut R_slice = res.fixed_view_mut::<3,3>(0,0);
    R_slice.copy_from(&R);

    let mut t_slice = res.fixed_view_mut::<3,1>(0,3);
    t_slice.copy_from(&t);

    res
}

//TODO: reuse exp_r in exp
#[allow(non_snake_case)]
pub fn exp_so3<F,T>(w: &Vector<F,U3,T>) -> Matrix3<F> where T: Storage<F,U3,U1>, F : GenericFloat {
    let omega = Float::sqrt((w.transpose()*w)[0]);

    match omega {
        omega if omega != F::zero() => {
            let omega_sqr = Float::powi(omega,2);
            let A = Float::sin(omega)/omega;
            let B = (F::one() - Float::cos(omega))/omega_sqr;
        
            let w_x = w.cross_matrix();
            let w_x_sqr = w_x*w_x;
            let I = Matrix3::<F>::identity();
            I + w_x.scale(A) + w_x_sqr.scale(B)
        },
        _ => Matrix3::identity()
    }
}


// Improvement: taylor expansion
#[allow(non_snake_case)]
pub fn ln_SO3<F, T>(R: &Matrix<F,U3,U3,T>) -> Matrix3<F> where T: Storage<F,U3,U3>, F : GenericFloat {
    let two: F = convert(2.0);
    let omega = Float::acos((R.trace() -F::one())/two);
    match omega {
        omega if omega != F::zero() => {
            let factor = omega/(two*Float::sin(omega));
            (R-R.transpose()).scale(factor)
        }
        _ => Matrix3::<F>::identity()
    }
}

#[allow(non_snake_case)]
pub fn ln<F>(se3: &Matrix4<F>) -> Vector6<F> where F :GenericFloat {
    let w_x = ln_SO3(&se3.fixed_view::<3,3>(0,0));
    let w_x_sqr = w_x*w_x;
    let two: F = convert(2.0);
    let w = vector_from_skew_symmetric(&w_x);
    let omega_sqrd = (w.transpose()*w)[0];
    let omega = Float::sqrt(omega_sqrd);
    let A = Float::sin(omega)/omega;
    let B = (F::one() - Float::cos(omega))/omega_sqrd;
    let factor = (F::one()-A/(two*B))/omega_sqrd;

    let I = Matrix3::<F>::identity();
    let V_inv = I-w_x.scale(convert(0.5)) +w_x_sqr.scale(factor);
    let u = V_inv*se3.fixed_view::<3,1>(0,3);

    let mut res = Vector6::<F>::zeros();
    let mut u_slice = res.fixed_view_mut::<3,1>(0,0);
    u_slice.copy_from(&u);
    let mut w_slice = res.fixed_view_mut::<3,1>(3,0);
    w_slice.copy_from(&w);

    res
}

/**
 * This is the jacobian of the rotation without a specific point being transformed
 */
#[allow(non_snake_case)]
pub fn right_jacobian<F,T>(w: &Vector<F,U3,T>) -> Matrix3<F> where T: Storage<F,U3,U1>, F : GenericFloat {
    let w_x = w.cross_matrix();
    let w_x_sqr = w_x*w_x;
    let w_norm = w.norm();
    let I = Matrix3::<F>::identity();

    match w_norm {
        w_norm if w_norm != F::zero() => {
            let w_norm_sqrd = Float::powi(w_norm,2);
            let w_norm_cubed = w_norm_sqrd*w_norm;
            let cos_norm = Float::cos(w_norm);
            let sin_norm = Float::sin(w_norm);
        
            let A = (F::one() - cos_norm)/(w_norm_sqrd);
            let B = (w_norm - sin_norm)/(w_norm_cubed);
                
            I - w_x.scale(A) + w_x_sqr.scale(B)
        },
        _ => I

    }
}

#[allow(non_snake_case)]
pub fn right_inverse_jacobian<F,T>(w: &Vector<F,U3,T>) -> Matrix3<F> where T: Storage<F,U3,U1>, F : GenericFloat {
    let w_x = w.cross_matrix();
    let w_x_sqr = w_x*w_x;
    let w_norm = w.norm();
    let I = Matrix3::<F>::identity();
    let two: F = convert(2.0);
    match w_norm {
        w_norm if w_norm != F::zero() => {
            let w_norm_sqrd = Float::powi(w_norm,2);
            let cos_norm = Float::cos(w_norm);
            let sin_norm = Float::sin(w_norm);
        
            let A = F::one()/w_norm_sqrd;
            let B = (F::one()+cos_norm)/(two*w_norm*sin_norm);
    
            I + w_x.scale(convert(0.5)) + w_x_sqr.scale(A+B)
        },
        _ => I

    }
}

pub fn chordal_distance<F>(a: &Matrix3<F>, b: &Matrix3<F>) -> F where F : GenericFloat {
    (a-b).norm()
}

pub fn angular_distance<F>(a: &Matrix3<F>) -> F where F: GenericFloat {
    ln_SO3(a).norm()
}



