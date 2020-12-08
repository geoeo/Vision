extern crate nalgebra as na;

use na::{U1,U3,Vector3,Quaternion,UnitQuaternion,Matrix4};
use crate::Float;

pub fn se3(t: &Vector3<Float>, quat: &Quaternion<Float>) -> Matrix4<Float> {
    let mut se3 = Matrix4::<Float>::identity();
    let unit_quat = UnitQuaternion::<Float>::from_quaternion(*quat); 
    let rot = unit_quat.to_rotation_matrix();

    let mut rot_slice = se3.fixed_slice_mut::<U3,U3>(0,0);
    rot_slice.copy_from(&(rot.matrix()));

    let mut trans_slice = se3.fixed_slice_mut::<U3,U1>(0,3);
    trans_slice.copy_from(&t);

    se3
}

pub fn invert_se3(pose: &Matrix4<Float>) -> Matrix4<Float> {
    let mut se3 = Matrix4::<Float>::identity();

    let mut rot_slice = se3.fixed_slice_mut::<U3,U3>(0,0);
    let pose_rot_slice = pose.fixed_slice::<U3,U3>(0,0);
    let pose_rot_transposed = pose_rot_slice.transpose();
    rot_slice.copy_from(&pose_rot_transposed);

    let mut trans_slice = se3.fixed_slice_mut::<U3,U1>(0,3);
    let pose_trans_slice = pose.fixed_slice::<U3,U1>(0,3);
    let pose_trans_inverted = -pose_rot_transposed*pose_trans_slice;
    trans_slice.copy_from(&pose_trans_inverted);

    se3
}

pub fn pose_difference(a: &Matrix4<Float>, b:&Matrix4<Float>) -> Matrix4<Float> {
    b*invert_se3(a)
}