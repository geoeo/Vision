extern crate nalgebra as na;

use na::{Vector3,SVector,Matrix3,Matrix4,Const, Vector, storage::Storage};
use crate::Float;


pub struct Bias {
    pub preintegration_jacobian_bias_g: Matrix3<Float>

}

impl Bias {

    //TODO:
    pub fn new(bias_accelerometer: &Vector3<Float>,gyro_delta_times: &Vec<Float>, delta_lie_i_k: &Vec<Vector3<Float>>, delta_rotations_i_k: &Vec<Matrix3::<Float>>) -> Bias {



        Bias {
            preintegration_jacobian_bias_g: Matrix3::<Float>::zeros()
        }
    }
}