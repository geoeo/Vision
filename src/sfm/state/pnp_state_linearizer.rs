extern crate nalgebra as na;
extern crate simba;

use std::collections::HashMap;
use na::{convert,Vector3,Vector2, DVector, Isometry3};
use crate::image::features::Feature;
use crate::sfm::state::{State,cam_state::{cam_extrinsic_state::CAMERA_PARAM_SIZE, cam_extrinsic_state::CameraExtrinsicState},landmark::{Landmark, euclidean_landmark::EuclideanLandmark}};
use crate::{Float,GenericFloat};
use crate::sensors::camera::Camera;

pub fn get_euclidean_landmark_state<F: GenericFloat, Feat: Feature,C1: Camera<Float>, C2: Camera<F>+Copy>(
    landmarks: &Vec<EuclideanLandmark<Float>>,
    camera_pose:  &Option<Isometry3<Float>>,
    camera: &C1
) -> State<F, C2, EuclideanLandmark<F>,CameraExtrinsicState<F,C2>,3, CAMERA_PARAM_SIZE> {
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
    let camera_2 = C2::from_matrices(&camera.get_projection().cast::<F>(), &camera.get_inverse_projection().cast::<F>());
    let cameras: Vec<C2> = vec![camera_2];
    State::new(initial_cam_pose, &cameras, euclidean_landmarks, &HashMap::from([(0, 0)]), 1, number_of_landmarks)
}

pub fn get_observed_features<F: GenericFloat, Feat: Feature>(features: &Vec<Feat>) -> DVector<F> {
    let mut observed_features = DVector::zeros(2*features.len());
    for (i,f) in features.iter().enumerate() {
        let orig = f.get_as_2d_point();
        let vals = Vector2::<F>::new(convert(orig.x),convert(orig.y));
        observed_features.fixed_view_mut::<2,1>(i*2,0).copy_from(&vals);
    }
    observed_features
}