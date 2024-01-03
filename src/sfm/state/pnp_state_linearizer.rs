extern crate nalgebra as na;
extern crate simba;

use simba::scalar::SubsetOf;
use std::collections::HashMap;
use na::{convert,Vector3,Vector2, DVector, Isometry3,base::Scalar, RealField};
use crate::image::features::Feature;
use crate::sfm::{state::{State,CAMERA_PARAM_SIZE}, landmark::{Landmark, euclidean_landmark::EuclideanLandmark}};
use crate::Float;

pub fn get_euclidean_landmark_state<F: Scalar + RealField + Copy + SubsetOf<Float>, Feat: Feature>(
    landmarks: &Vec<EuclideanLandmark<Float>>,
    camera_pose:  &Option<Isometry3<Float>>
) -> State<F, EuclideanLandmark<F>,3> {
    let euclidean_landmarks = landmarks.iter().map(|l| {
        let id = l.get_id();
        assert!(id.is_some());
        let v = l.get_state_as_vector();
        EuclideanLandmark::from_state_with_id(Vector3::<F>::new(
        convert(v.x),
        convert(v.y),
        convert(v.z)
    ),&id)
}).collect::<Vec<_>>();
    let initial_cam_pose_iso = match camera_pose {
        Some(pose) => convert::<Isometry3<Float>,Isometry3<F>>(pose.clone()),
        None => Isometry3::<F>::identity()
    };
    let mut initial_cam_pose = DVector::<F>::zeros(CAMERA_PARAM_SIZE);
    initial_cam_pose.fixed_view_mut::<3,1>(0,0).copy_from(&initial_cam_pose_iso.translation.vector);
    initial_cam_pose.fixed_view_mut::<3,1>(3,0).copy_from(&initial_cam_pose_iso.rotation.scaled_axis());


    let number_of_landmarks = euclidean_landmarks.len();
    State::new(initial_cam_pose, euclidean_landmarks, &HashMap::from([(0, 0)]), 1, number_of_landmarks)
}

pub fn get_observed_features<F: Scalar + RealField, Feat: Feature>(features: &Vec<Feat>) -> DVector<F> {
    let mut observed_features = DVector::zeros(2*features.len());
    for (i,f) in features.iter().enumerate() {
        let orig = f.get_as_2d_point();
        let vals = Vector2::<F>::new(convert(orig.x),convert(orig.y));
        observed_features.fixed_view_mut::<2,1>(i*2,0).copy_from(&vals);
    }
    observed_features
}