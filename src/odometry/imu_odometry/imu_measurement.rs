extern crate nalgebra as na;

use na::{Vector3,Matrix3,MatrixN,Matrix6,U9};
use crate::Float;

pub type ImuCovariance = MatrixN<Float,U9>;
pub type NoiseCovariance = Matrix6<Float>;

pub struct ImuState {
    pub position: Vector3<Float>,
    pub velocity: Vector3<Float>,
    pub orientation: Matrix3<Float>
}