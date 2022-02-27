extern crate nalgebra as na;

use na::{Vector3,Matrix4,Matrix3,Point3,UnitQuaternion, Isometry3, Translation3};
use crate::Float;

pub fn from_matrix(mat: &Matrix4<Float>) -> Isometry3<Float> {
    let vec = Vector3::<Float>::new(mat[(0,3)],mat[(1,3)],mat[(2,3)]);
    let rot = Matrix3::<Float>::new(mat[(0,0)],mat[(0,1)],mat[(0,2)],
                                    mat[(1,0)],mat[(1,1)],mat[(1,2)],
                                    mat[(2,0)],mat[(2,1)],mat[(2,2)]);

    Isometry3::<Float>::from_parts(Translation3::from(vec), UnitQuaternion::<Float>::from_matrix(&rot))
}

pub fn se3(t: &Vector3<Float>, quat: &UnitQuaternion<Float>) -> Isometry3<Float> {
    Isometry3::<Float>::from_parts(Translation3::from(*t), *quat)
}

/**
 * Transform from a to b
 */
pub fn pose_difference(a: &Isometry3<Float>, b:&Isometry3<Float>) -> Isometry3<Float> {
    b*a.inverse()
}

pub fn decomp(pose:&Isometry3<Float>) -> (Vector3<Float>,Matrix3<Float>) {
    (pose.translation.vector,pose.rotation.to_rotation_matrix().matrix().into_owned())
}

pub fn apply_pose_deltas_to_point(point: Point3<Float>, pose_deltas: &Vec<Isometry3<Float>>) -> Vec<Point3<Float>> {
    pose_deltas.iter().scan(point, |acc, &pose_delta| {
        *acc = pose_delta*(*acc);
        Some(*acc)
    }).collect::<Vec<Point3<Float>>>()
}

// Error according to A Benchmark for the Evaluation of RGB-D SLAM Systems
pub fn error(q_1: &Isometry3<Float>,q_2: &Isometry3<Float>,p_1: &Isometry3<Float>,p_2: &Isometry3<Float>) -> Isometry3<Float> {
    ((q_1.inverse()*q_2.inverse())*(p_1.inverse()*p_2.inverse())).inverse()
}

pub fn rsme(data: &Vec<Isometry3<Float>>) -> Float {
    let norm_sum = data.iter().fold(0.0, |acc, x| acc + x.translation.vector.norm_squared());
    (norm_sum/(data.len() as Float)).sqrt()
}