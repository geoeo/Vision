extern crate nalgebra as na;

use na::{U1,U3,Vector3,Vector4,Quaternion,UnitQuaternion,Matrix4};
use crate::Float;

pub fn se3(t: &Vector3<Float>, quat: &Quaternion<Float>) -> Matrix4<Float> {
    let mut se3 = Matrix4::<Float>::identity();
    let unit_quat = UnitQuaternion::<Float>::from_quaternion(*quat); 
    let rot = unit_quat.to_rotation_matrix();

    let mut rot_slice = se3.fixed_slice_mut::<3,3>(0,0);
    rot_slice.copy_from(&(rot.matrix()));

    let mut trans_slice = se3.fixed_slice_mut::<3,1>(0,3);
    trans_slice.copy_from(&t);

    se3
}

pub fn invert_se3(pose: &Matrix4<Float>) -> Matrix4<Float> {
    let mut se3 = Matrix4::<Float>::identity();

    let mut rot_slice = se3.fixed_slice_mut::<3,3>(0,0);
    let pose_rot_slice = pose.fixed_slice::<3,3>(0,0);
    let pose_rot_transposed = pose_rot_slice.transpose();
    rot_slice.copy_from(&pose_rot_transposed);

    let mut trans_slice = se3.fixed_slice_mut::<3,1>(0,3);
    let pose_trans_slice = pose.fixed_slice::<3,1>(0,3);
    let pose_trans_inverted = -pose_rot_transposed*pose_trans_slice;
    trans_slice.copy_from(&pose_trans_inverted);

    se3
}

pub fn pose_difference(a: &Matrix4<Float>, b:&Matrix4<Float>) -> Matrix4<Float> {
    b*invert_se3(a)
}

pub fn apply_pose_deltas_to_point(point: Vector4<Float>, pose_deltas: &Vec<Matrix4<Float>>) -> Vec<Vector4<Float>> {
    pose_deltas.iter().scan(point, |acc, &pose_delta| {
        *acc = pose_delta*(*acc);
        Some(*acc)
    }).collect::<Vec<Vector4<Float>>>()
}

// Error according to A Benchmark for the Evaluation of RGB-D SLAM Systems
pub fn error(q_1: &Matrix4<Float>,q_2: &Matrix4<Float>,p_1: &Matrix4<Float>,p_2: &Matrix4<Float>) -> Matrix4<Float> {
    invert_se3(&(invert_se3(q_1)*q_2))*(invert_se3(p_1)*p_2)
}

pub fn rsme(data: &Vec<Matrix4<Float>>) -> Float {
    let norm_sum = data.iter().fold(0.0, |acc, x| acc + x.fixed_slice::<3,1>(0,3).norm_squared());
    (norm_sum/(data.len() as Float)).sqrt()
}