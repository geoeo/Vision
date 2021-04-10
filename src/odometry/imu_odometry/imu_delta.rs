extern crate nalgebra as na;

use na::{Vector3,Matrix3,Matrix4,U1,U3};
use crate::Float;



pub struct ImuDelta {
    pub delta_position: Vector3<Float>,
    pub delta_velocity: Vector3<Float>,
    pub delta_rotation_i_k: Matrix3<Float>,
    pub delta_rotation_k: Matrix3<Float>
}

impl ImuDelta {

    pub fn empty() -> ImuDelta {
        ImuDelta {
            delta_position: Vector3::<Float>::zeros(),
            delta_velocity: Vector3::<Float>::zeros(),
            delta_rotation_i_k: Matrix3::<Float>::identity(),
            delta_rotation_k: Matrix3::<Float>::identity()
        }
    }

    pub fn delta_rotation(&self) -> Matrix3<Float> {
        self.delta_rotation_i_k*self.delta_rotation_k
    }

    pub fn get_pose(&self) -> Matrix4<Float> {
        let mut pose = Matrix4::<Float>::identity();
        pose.fixed_slice_mut::<U3,U3>(0,0).copy_from(&self.delta_rotation());
        pose.fixed_slice_mut::<U3,U1>(0,3).copy_from(&self.delta_position);
        pose
    }
}