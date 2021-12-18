extern crate nalgebra as na;

use na::{Vector3,Point3,Quaternion,UnitQuaternion, Isometry3, Translation3};
use crate::Float;

pub fn se3(t: &Vector3<Float>, quat: &Quaternion<Float>) -> Isometry3<Float> {
    let unit_quat = UnitQuaternion::<Float>::from_quaternion(*quat); 
    Isometry3::<Float>::from_parts(Translation3::from(*t), unit_quat)
}

pub fn pose_difference(a: &Isometry3<Float>, b:&Isometry3<Float>) -> Isometry3<Float> {
    b*a.inverse()
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

//TODO: inverse depth param conversion 