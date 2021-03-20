extern crate nalgebra as na;

use na::{Vector3,Matrix3,MatrixN,Matrix3x2,U9};
use crate::Float;


pub type ImuCovariance = MatrixN<Float,U9>;
pub type NoiseCovariance = Matrix3x2<Float>;

pub struct ImuState {
    pub position: Vector3<Float>,
    pub velocity: Vector3<Float>,
    pub orientation: Matrix3<Float>
}