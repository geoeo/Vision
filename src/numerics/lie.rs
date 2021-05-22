extern crate nalgebra as na;

use na::{Vector,Vector3,Vector6,Matrix3,Matrix3x6,Matrix, Matrix4,U3,U1,base::storage::Storage};
use crate::Float;

pub fn skew_symmetric<T>(w: &Vector<Float,U3,T>) -> Matrix3<Float> where T: Storage<Float,U3,U1>  {
    Matrix3::<Float>::new(0.0, -w[2], w[1],
                          w[2], 0.0, -w[0],
                          -w[1], w[0], 0.0)
}

pub fn vector_from_skew_symmetric(w_x: &Matrix3<Float>) -> Vector3<Float> {
    Vector3::<Float>::new(w_x[(2,1)],w_x[(0,2)],w_x[(1,0)])
}

pub fn left_jacobian_around_identity<T>(transformed_position: &Vector<Float,U3,T>) -> Matrix3x6<Float> where T: Storage<Float,U3,U1> {
    let skew_symmetrix = skew_symmetric(&(-1.0*transformed_position));
    let mut jacobian = Matrix3x6::<Float>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    
    for i in 3..6 {
        jacobian.set_column(i, &skew_symmetrix.column(i-3));
    }

    jacobian

}

//TODO: check this
pub fn right_jacobian_around_identity<T>(position: &Vector<Float,U3,T>, rotation: &Matrix3<Float>) -> Matrix3x6<Float> where T: Storage<Float,U3,U1> {
    let skew_symmetrix = skew_symmetric(&(-1.0*rotation*position))*rotation;
    let mut jacobian = Matrix3x6::<Float>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

    for i in 3..6 {
        jacobian.set_column(i, &skew_symmetrix.column(i-3));
    }

    jacobian

} 

//TODO: taylor expansion, check if this is correct
#[allow(non_snake_case)]
pub fn exp<T>(u: &Vector<Float,U3,T>, w: &Vector<Float,U3,T>) -> Matrix4<Float> where T: Storage<Float,U3,U1> {
    let omega = (w.transpose()*w)[0].sqrt();
    let omega_sqr = omega.powi(2);
    let A = omega.sin()/omega;
    let B = (1.0 - omega.cos())/omega_sqr;
    let C = (1.0 - A)/omega_sqr;


    let w_x = skew_symmetric(w);
    let w_x_sqr = w_x*w_x;
    let I = Matrix3::<Float>::identity();
    let R = I + A*w_x + B*w_x_sqr;
    let V = I + B*w_x + C*w_x_sqr;
    let t = V*u;

    let mut res = Matrix4::<Float>::identity();
    let mut R_slice = res.fixed_slice_mut::<3,3>(0,0);
    R_slice.copy_from(&R);

    let mut t_slice = res.fixed_slice_mut::<3,1>(0,3);
    t_slice.copy_from(&t);

    res

}

//TODO: reuse exp_r in exp
#[allow(non_snake_case)]
pub fn exp_r<T>(w: &Vector<Float,U3,T>) -> Matrix3<Float> where T: Storage<Float,U3,U1> {
    let omega = (w.transpose()*w)[0].sqrt();

    match omega {
        omega if omega != 0.0 => {
            let omega_sqr = omega.powi(2);
            let A = omega.sin()/omega;
            let B = (1.0 - omega.cos())/omega_sqr;
            //let C = (1.0 - A)/omega_sqr;
        
            let w_x = skew_symmetric(w);
            let w_x_sqr = w_x*w_x;
            let I = Matrix3::<Float>::identity();
            I + A*w_x + B*w_x_sqr
        },
        _ => Matrix3::identity()
    }
}


// TODO: taylor expansion, check this
#[allow(non_snake_case)]
pub fn ln_SO3<T>(R: &Matrix<Float,U3,U3,T>) -> Matrix3<Float> where T: Storage<Float,U3,U3> {
    let omega = ((R.trace() -1.0)/2.0).acos();
    match omega {
        omega if omega != 0.0 => {
            let factor = omega/(2.0*omega.sin());
            factor*(R-R.transpose())
        }
        _ => Matrix3::identity()
    }
}

#[allow(non_snake_case)]
pub fn ln(se3: &Matrix4<Float>) -> Vector6<Float> {
    let w_x = ln_SO3(&se3.fixed_slice::<3,3>(0,0));
    let w_x_sqr = w_x*w_x;
    let w = vector_from_skew_symmetric(&w_x);
    let omega_sqr = (w.transpose()*w)[0];
    let omega = omega_sqr.sqrt();
    let A = omega.sin()/omega;
    let B = (1.0 - omega.cos())/omega_sqr;
    let factor = (1.0-A/(2.0*B))/omega_sqr;

    let I = Matrix3::<Float>::identity();
    let V_inv = I-0.5*w_x +factor*w_x_sqr;
    let u = V_inv*se3.fixed_slice::<3,1>(0,3);

    let mut res = Vector6::<Float>::zeros();
    let mut u_slice = res.fixed_slice_mut::<3,1>(0,0);
    u_slice.copy_from(&u);
    let mut w_slice = res.fixed_slice_mut::<3,1>(3,0);
    w_slice.copy_from(&w);

    res
}


#[allow(non_snake_case)]
pub fn right_jacobian<T>(w: &Vector<Float,U3,T>) -> Matrix3<Float> where T: Storage<Float,U3,U1> {
    let w_x = skew_symmetric(&w);
    let w_x_sqr = w_x*w_x;
    let w_norm = w.norm();
    let I = Matrix3::<Float>::identity();

    match w_norm {
        w_norm if w_norm != 0.0 => {
            let w_norm_sqrd = w_norm.powi(2);
            let w_norm_cubed = w_norm_sqrd*w_norm;
            let cos_norm = w_norm.cos();
            let sin_norm = w_norm.sin();
        
            let A = (1.0 - cos_norm)/(w_norm_sqrd);
            let B = (w_norm - sin_norm)/(w_norm_cubed);
                
            I - A*w_x + B*w_x_sqr
        },
        _ => I

    }
}

#[allow(non_snake_case)]
pub fn right_inverse_jacobian<T>(w: &Vector<Float,U3,T>) -> Matrix3<Float> where T: Storage<Float,U3,U1> {
    let w_x = skew_symmetric(&w);
    let w_x_sqr = w_x*w_x;
    let w_norm = w.norm();
    let I = Matrix3::<Float>::identity();

    match w_norm {
        w_norm if w_norm != 0.0 => {
            let w_norm_sqrd = w_norm.powi(2);
            let cos_norm = w_norm.cos();
            let sin_norm = w_norm.sin();
        
            let A = 1.0/w_norm_sqrd;
            let B = (1.0+cos_norm)/(2.0*w_norm*sin_norm);
    
            I + 0.5*w_x + (A+B)*w_x_sqr
        },
        _ => I

    }
}



