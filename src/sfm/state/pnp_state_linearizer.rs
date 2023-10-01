extern crate nalgebra as na;
extern crate num_traits;

use na::{convert,Vector3,Vector2, Matrix4xX, DVector, Isometry3,base::Scalar, RealField};
use num_traits::float;
use simba::scalar::SupersetOf;
use crate::image::features::Feature;
use crate::sfm::{state::{State,CAMERA_PARAM_SIZE}, landmark::{Landmark, euclidean_landmark::EuclideanLandmark}};
use crate::Float;

pub fn get_euclidean_landmark_state<F: float::Float + Scalar + RealField + SupersetOf<Float>, Feat: Feature>(
    landmarks: &Matrix4xX<Float>,
    camera_pose:  &Option<Isometry3<Float>>
) -> State<F, EuclideanLandmark<F>,3> {
    let euclidean_landmarks = landmarks.column_iter().map(|c| EuclideanLandmark::from_state(Vector3::<F>::new(
        convert(c[0]),
        convert(c[1]),
        convert(c[2])
    ))).collect::<Vec<_>>();
    let initial_cam_pose_iso = match camera_pose {
        Some(pose) => convert::<Isometry3<Float>,Isometry3<F>>(pose.clone()),
        None => Isometry3::<F>::identity()
    };
    let mut initial_cam_pose = DVector::<F>::zeros(CAMERA_PARAM_SIZE);
    initial_cam_pose.fixed_view_mut::<3,1>(0,0).copy_from(&initial_cam_pose_iso.translation.vector);
    initial_cam_pose.fixed_view_mut::<3,1>(3,0).copy_from(&initial_cam_pose_iso.rotation.scaled_axis());

    let number_of_landmarks = euclidean_landmarks.len();
    State::new(initial_cam_pose, euclidean_landmarks, 1, number_of_landmarks)
}

pub fn get_observed_features<F: float::Float + Scalar + RealField + SupersetOf<Float>, Feat: Feature>(features: &Vec<Feat>) -> DVector<F> {
    let mut observed_features = DVector::zeros(2*features.len());
    for (i,f) in features.iter().enumerate() {
        let orig = f.get_as_2d_point();
        let vals = Vector2::<F>::new(convert(orig.x),convert(orig.y));
        observed_features.fixed_view_mut::<2,1>(i*2,1).copy_from(&vals);
    }
    observed_features
}