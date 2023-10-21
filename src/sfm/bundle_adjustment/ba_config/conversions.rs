extern crate nalgebra as na;
extern crate num_traits;

use na::{Isometry3, Matrix4xX};
use std::collections::{HashMap, HashSet};
use crate::image::features::{matches::Match, Feature};
use crate::sfm::landmark::{Landmark, euclidean_landmark::EuclideanLandmark};
use crate::Float;



pub fn compute_absolute_poses_for_root(
    root: usize,
    paths: &Vec<Vec<(usize, usize)>>,
    pose_map: &HashMap<(usize, usize), Isometry3<Float>>,
) -> HashMap<usize, Isometry3<Float>> {
    let flattened_path_len = paths.iter().flatten().collect::<Vec<_>>().len();
    let mut abs_pose_map =
        HashMap::<usize, Isometry3<Float>>::with_capacity(flattened_path_len + 1);
    abs_pose_map.insert(root, Isometry3::<Float>::identity());

    for path in paths {
        let mut pose_acc = Isometry3::<Float>::identity();
        abs_pose_map.insert(path[0].0, pose_acc);
        for key in path {
            let pose = pose_map
                .get(key)
                .expect("Error in compute_absolute_poses_for_root: Pose for key not found ");
            pose_acc *= pose;
            abs_pose_map.insert(key.1, pose_acc);
        }
    }
    abs_pose_map
}

fn compute_absolute_landmarks_for_root(
    paths: &Vec<(usize, usize)>,
    landmark_map: &HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>,
    abs_pose_map: &HashMap<usize, Isometry3<Float>>
) -> HashMap<(usize,usize), Matrix4xX<Float>> {
    let mut abs_landmark_map =
        HashMap::<(usize,usize), Matrix4xX<Float>>::with_capacity(paths.len());
    for (id_s, id_f) in paths {
        let landmark_key = (*id_s, *id_f);
        let landmarks = landmark_map.get(&landmark_key).expect(format!("Landmark missing for key {:?}",landmark_key).as_str());
        let triangulated_matches = generate_landmark_matrix(landmarks);
        let abs_pose_w_s = abs_pose_map
            .get(id_s)
            .expect("compute_absolute_landmarks_for_root: abs pose not found")
            .to_matrix();
        let root_aligned_triangulated_matches = abs_pose_w_s * &triangulated_matches;
        abs_landmark_map.insert(landmark_key, root_aligned_triangulated_matches);
    }
    abs_landmark_map
}

pub fn compute_features_per_image_map<Feat: Feature + Clone>(
    match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>,
    unique_landmark_ids: &HashSet<usize>,
    unique_camera_ids: &Vec<usize>,
) -> HashMap<usize, Vec<Feat>> {
    let mut feature_map = HashMap::<usize, Vec<Feat>>::with_capacity(unique_camera_ids.len());
    for unique_cam_id in unique_camera_ids {
        feature_map.insert(
            unique_cam_id.clone(),
            Vec::<Feat>::with_capacity(unique_landmark_ids.len()),
        );
    }

    for ((cam_id_1, cam_id_2), matches) in match_map.iter() {
        for vec_idx in 0..matches.len() {
            let m = &matches[vec_idx];
            let f_1 = m.get_feature_one();
            let f_2 = m.get_feature_two();
            feature_map
                .get_mut(cam_id_1)
                .expect("compute_features_per_image_map: Camera not found in bck feature")
                .push(f_1.clone());
            feature_map
                .get_mut(cam_id_2)
                .expect("compute_features_per_image_map: Camera not found in bck feature")
                .push(f_2.clone());
        }
    }

    feature_map
}

pub fn compute_path_pairs_as_vec(root: usize, paths: &Vec<Vec<usize>>) -> Vec<Vec<(usize, usize)>> {
    let number_of_paths = paths.len();
    let mut all_path_pairs = Vec::<Vec<(usize, usize)>>::with_capacity(number_of_paths);
    for path_idx in 0..number_of_paths {
        let path = &paths[path_idx];
        let mut path_pair = Vec::<(usize, usize)>::with_capacity(path.len());
        for j in 0..path.len() {
            let id1 = match j {
                0 => root,
                idx => path[idx - 1],
            };
            let id2 = path[j];
            path_pair.push((id1, id2));
        }
        all_path_pairs.push(path_pair);
    }
    all_path_pairs
}

pub fn compute_path_id_pairs(root_id: usize, paths: &Vec<Vec<usize>>) -> Vec<Vec<(usize, usize)>> {
    let mut path_id_paris = Vec::<Vec<(usize, usize)>>::with_capacity(paths.len());
    for sub_path in paths {
        path_id_paris.push(
            sub_path
                .iter()
                .enumerate()
                .map(|(i, &id)| match i {
                    0 => (root_id, id),
                    idx => (sub_path[idx - 1], id),
                })
                .collect(),
        )
    }
    path_id_paris
}

/**
 * With respect to the root camera
 */
pub fn generate_abs_landmark_map(root: usize, paths: &Vec<Vec<usize>>, 
    landmark_map: &HashMap<(usize,usize),Vec<EuclideanLandmark<Float>>>, 
    abs_pose_map: &HashMap<usize, Isometry3<Float>>) -> HashMap<(usize, usize), Matrix4xX<Float>> {
    let path_id_pairs = compute_path_id_pairs(root, paths);
    compute_absolute_landmarks_for_root(&path_id_pairs.into_iter().flatten().collect(), landmark_map, abs_pose_map)
}

pub fn generate_landmark_matrix(landmarks: &Vec<EuclideanLandmark<Float>>) -> Matrix4xX<Float> {
    let mut mat = Matrix4xX::<Float>::from_element(landmarks.len(), 1.0);
    for i in 0..landmarks.len() {
        let vec = landmarks[i].get_state_as_vector();
        mat.fixed_view_mut::<3,1>(0,i).copy_from(&vec);
    }
    mat
}