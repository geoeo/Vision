extern crate nalgebra as na;

use na::{Vector3,Matrix4,Matrix3,Point3,UnitQuaternion,Isometry ,Isometry3, Translation3, Rotation3, convert, SimdRealField, ComplexField,base::Scalar, RealField};
use num_traits::{float,NumAssign};

pub fn from_matrix<F>(mat: &Matrix4<F>) -> Isometry3<F> where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    let vec = Vector3::<F>::new(mat[(0,3)],mat[(1,3)],mat[(2,3)]);
    let rot = Matrix3::<F>::new(mat[(0,0)],mat[(0,1)],mat[(0,2)],
                                    mat[(1,0)],mat[(1,1)],mat[(1,2)],
                                    mat[(2,0)],mat[(2,1)],mat[(2,2)]);

    Isometry3::<F>::from_parts(Translation3::from(vec), UnitQuaternion::<F>::from_matrix(&rot))
}


pub fn se3<F>(t: &Vector3<F>, rotation: &Matrix3<F>) -> Matrix4<F> where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField{
    Isometry::<F, Rotation3<F>,3>::from_parts(Translation3::from(*t),Rotation3::from_matrix_eps(rotation, convert(2e-16), 100, Rotation3::identity())).to_homogeneous()
}

pub fn from_parts<F>(t: &Vector3<F>, quat: &UnitQuaternion<F>) -> Isometry3<F> where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    Isometry3::<F>::from_parts(Translation3::from(*t), *quat)
}

pub fn to_parts<F>(isometry:  &Isometry3::<F>) -> (Vector3<F>, Matrix3<F>) where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    let mat = isometry.to_matrix();
    let translation = Vector3::<F>::new(mat[(0,3)],mat[(1,3)],mat[(2,3)]);
    let rot = Matrix3::<F>::new(mat[(0,0)],mat[(0,1)],mat[(0,2)],
                                    mat[(1,0)],mat[(1,1)],mat[(1,2)],
                                    mat[(2,0)],mat[(2,1)],mat[(2,2)]);
    (translation, rot)
}

/**
 * Transform from a to b
 */
pub fn pose_difference<F>(a: &Isometry3<F>, b:&Isometry3<F>) -> Isometry3<F> where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    b*a.inverse()
}

pub fn decomp<F>(pose:&Isometry3<F>) -> (Vector3<F>,Matrix3<F>) where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    (pose.translation.vector,pose.rotation.to_rotation_matrix().matrix().into_owned())
}

pub fn apply_pose_deltas_to_point<F>(point: Point3<F>, pose_deltas: &Vec<Isometry3<F>>) -> Vec<Point3<F>> where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    pose_deltas.iter().scan(point, |acc, &pose_delta| {
        *acc = pose_delta*(*acc);
        Some(*acc)
    }).collect::<Vec<Point3<F>>>()
}

// Error according to A Benchmark for the Evaluation of RGB-D SLAM Systems
pub fn error<F>(q_1: &Isometry3<F>,q_2: &Isometry3<F>,p_1: &Isometry3<F>,p_2: &Isometry3<F>) -> Isometry3<F> where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    ((q_1.inverse()*q_2.inverse())*(p_1.inverse()*p_2.inverse())).inverse()
}

pub fn rsme<F>(data: &Vec<Isometry3<F>>) -> F where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField{
    let norm_sum = data.iter().fold(F::zero(), |acc, x| acc + convert(x.translation.vector.norm_squared()));
    let length = data.len() as f64;
    float::Float::sqrt(norm_sum/convert(length))
}

/**
 * 3D Rotations - Kanatani p.35
 */
pub fn optimal_correction_of_rotation<F>(rotation: &Matrix3<F>) -> Matrix3<F> where F : float::Float + Scalar + NumAssign + RealField + SimdRealField + ComplexField {
    let mut svd = rotation.svd(true,true);
    let u = &svd.u.expect("optimal_correction_of_rotation: SVD failed on u");
    let v_t = &svd.v_t.expect("optimal_correction_of_rotation: SVD failed on v_t");
    svd.singular_values[0] = F::one();
    svd.singular_values[1] = F::one();
    svd.singular_values[2] = (u*v_t.transpose()).determinant();
    svd.recompose().expect("optimal_correction_of_rotation: SVD failed on recompose")
}