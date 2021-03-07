extern crate nalgebra as na;

use na::{Vector3,Matrix3,MatrixN,U9};
use crate::Float;


pub type ImuCovariance =  MatrixN<Float,U9>;

pub struct ImuState {
    pub position: Vector3<Float>,
    pub velocity: Vector3<Float>,
    pub orientation: Matrix3<Float>
}