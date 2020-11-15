extern crate nalgebra as na;

use na::{Vector3,Matrix3,Matrix3x6};
use crate::Float;

pub fn skew_symmetric(vec: &Vector3<Float>) -> Matrix3<Float> {
    Matrix3::<Float>::new(0.0, -vec[2], vec[1],
                          vec[2], 0.0, -vec[0],
                          -vec[1], vec[0], 0.0)
}

pub fn jacobian_with_respect_to_rotation(position: &Vector3<Float>) -> Matrix3x6<Float> {
    let skew_symmetrix = skew_symmetric(&position);
    let mut jacobian = Matrix3x6::<Float>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    
    for i in 3..6 {
        jacobian.set_column(i, &skew_symmetrix.column(i));
    }

    jacobian

} 