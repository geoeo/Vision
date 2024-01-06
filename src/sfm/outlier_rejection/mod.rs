extern crate nalgebra as na;

use na::{DVector,Matrix3x4,Matrix4xX};
use crate::image::features::{matches::Match,Feature};
use crate::sensors::camera::Camera;
use crate::{float,Float};
use std::{collections::{HashMap,HashSet}, marker::{Send,Sync}};

use super::landmark::euclidean_landmark::EuclideanLandmark;

pub mod dual;

pub fn calculate_reprojection_errors<Feat: Feature + Send + Sync, C: Camera<Float>>(landmarks: &Matrix4xX<Float>, matches: &Vec<Match<Feat>>, transform_c1: &Matrix3x4<Float>, cam_1 :&C, transform_c2: &Matrix3x4<Float>, cam_2 :&C) -> DVector<Float> {
    let landmark_count = landmarks.ncols();
    let mut reprojection_errors = DVector::<Float>::zeros(landmark_count);

    for i in 0..landmarks.ncols() {
        let p = landmarks.fixed_columns::<1>(i).into_owned();
        let m = &matches[i];
        let feat_1 = &m.get_feature_one().get_as_2d_point();
        let feat_2 = &m.get_feature_two().get_as_2d_point();
        let p_cam_1 = transform_c1*p;
        let p_cam_2 = transform_c2*p;

        let projected_1 = cam_1.project(&p_cam_1);
        let projected_2 = cam_2.project(&p_cam_2);

        if projected_1.is_some() && projected_2.is_some() {
            let projected_1 = projected_1.unwrap().to_vector();
            let projected_2 = projected_2.unwrap().to_vector();
            reprojection_errors[i] = (feat_1-projected_1).norm() + (feat_2-projected_2).norm()
        } else {
            reprojection_errors[i] = float::INFINITY;
        }
    }
    reprojection_errors
}

pub fn reject_matches_via_disparity<Feat: Feature + Clone + Send + Sync>(disparitiy_map: HashMap<(usize, usize),DVector<Float>>, match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>, disparity_cutoff: Float) {
    let keys = match_map.keys().map(|key| *key).collect::<Vec<_>>();
    for key in &keys {
        let disparities = disparitiy_map.get(key).unwrap();
        let matches = match_map.get(key).unwrap();
        assert_eq!(disparities.nrows(), matches.len());
        let filtered_matches = matches.iter().enumerate().filter(|&(idx,_)| disparities[idx] >= disparity_cutoff).map(|(_,v)| v.clone()).collect::<Vec<Match<Feat>>>();
        match_map.insert(*key,filtered_matches);
    }
}

pub fn calcualte_disparities<Feat: Feature + Send + Sync>(matches: &Vec<Match<Feat>>) -> DVector<Float> {
    let mut disparities = DVector::<Float>::zeros(matches.len());
    for i in 0..matches.len() {
        let m = &matches[i];
        disparities[i] = (m.get_feature_one().get_as_2d_point() - m.get_feature_two().get_as_2d_point()).norm()
    }
    disparities
}

pub fn reject_landmark_outliers<Feat: Feature + Clone + Send + Sync>(
    landmark_map: &mut  HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>, 
    reprojection_error_map: &mut HashMap<(usize, usize),DVector<Float>>, 
    match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
    match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
    first_landmark_sighting_map: &mut HashMap<usize, usize>,
    landmark_cutoff: Float){
        let keys = match_norm_map.keys().map(|key| *key).collect::<Vec<_>>();
        let mut rejected_landmark_ids = HashSet::<usize>::with_capacity(1000);

        for key in &keys {
            let reprojection_erros = reprojection_error_map.get(key).unwrap();
            let matches = match_norm_map.get(key).unwrap();

            let rejected_indices = reprojection_erros.iter().enumerate().filter(|&(_,v_reporj)| *v_reporj >= landmark_cutoff).map(|(idx,_)| idx).collect::<HashSet<usize>>();
            rejected_landmark_ids.extend(matches.iter().enumerate().filter(|(idx,_)|rejected_indices.contains(idx)).map(|(_,v)| v.get_landmark_id().unwrap()).collect::<HashSet<_>>());
        }

        for key in &keys {
            let reprojection_erros = reprojection_error_map.get(key).unwrap();
            let matches_norm = match_norm_map.get(key).unwrap();
            let matches = match_map.get(key).unwrap();
            let landmarks = landmark_map.get(key).unwrap();

            let accepted_indices = matches_norm.iter().enumerate().filter(|&(_,v)| !rejected_landmark_ids.contains(&v.get_landmark_id().unwrap())).map(|(idx,_)| idx).collect::<HashSet<usize>>();

            let filtered_matches_norm = matches_norm.iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| v.clone()).collect::<Vec<Match<Feat>>>();
            let filtered_matches = matches.iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| v.clone()).collect::<Vec<Match<Feat>>>();
            assert!(!&filtered_matches_norm.is_empty(), "reject outliers empty features for : {:?}", key);

            let filtered_landmarks = landmarks.iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| v.clone()).collect::<Vec<_>>();
            assert!(!&filtered_landmarks.is_empty());
        
            let filtered_reprojection_errors_vec = reprojection_erros.iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| *v).collect::<Vec<Float>>();
            assert!(!&filtered_reprojection_errors_vec.is_empty());
            let filtered_reprojection_errors = DVector::<Float>::from_vec(filtered_reprojection_errors_vec);

            match_norm_map.insert(*key,filtered_matches_norm);
            match_map.insert(*key,filtered_matches);
            reprojection_error_map.insert(*key,filtered_reprojection_errors);
            landmark_map.insert(*key,filtered_landmarks);
        }

        for rejected_id in rejected_landmark_ids {
            first_landmark_sighting_map.remove(&rejected_id);
        }
        assert!(!&first_landmark_sighting_map.is_empty())

}

pub fn filter_by_rejected_landmark_ids<Feat: Feature + Clone + Send + Sync>(
    rejected_landmark_ids: &HashSet<usize>,
    match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>, 
    match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
    landmark_map: &mut HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>, 
    reprojection_error_map: &mut HashMap<(usize, usize),DVector<Float>>,
    first_landmark_sighting_map: &mut HashMap<usize, usize>
) -> () {

    // Update match map, match_norm_map, landmark_map, abs_landmark_map -> group 1
    let cam_pairs = match_norm_map.keys().map(|(id1,id2)| (*id1,*id2)).collect::<Vec<_>>();
    for cam_key in cam_pairs {
        let matches_norm = match_norm_map.get(&cam_key).expect("filter_by_rejected_landmark_ids: matches norm, missing cam pair");
        let matches = match_map.get(&cam_key).expect("filter_by_rejected_landmark_ids: matches, missing cam pair");
        let landmarks = landmark_map.get(&cam_key).expect("filter_by_rejected_landmark_ids: landmarks, missing cam pair");
        let reprojections = reprojection_error_map.get(&cam_key).expect("filter_by_rejected_landmark_ids: reprojection_error_map, missing cam pair");

        let (match_indices_filtered, matches_norm_filtered) : (HashSet<_>, Vec<_>) 
            = matches_norm.iter().enumerate().filter(|(_, m)| !rejected_landmark_ids.contains(&m.get_landmark_id().expect("filter_by_rejected_landmark_ids: no landmark it for filtering"))).map(|(i,m)| (i, m.clone())).unzip();
        let matches_filtered : Vec<_> 
            = matches.iter().filter(|m| !rejected_landmark_ids.contains(&m.get_landmark_id().expect("filter_by_rejected_landmark_ids: no landmark it for filtering"))).map(|m| m.clone()).collect();

        let landmarks_filtered : Vec<_> = landmarks.iter().enumerate().filter(|(i,_)| match_indices_filtered.contains(i)).map(|(_,c)| c.clone()).collect();

        assert!(!landmarks_filtered.is_empty(), "assertion failed: !landmarks_filtered_as_vec.is_empty() - for cam pair {:?}",cam_key);
        let reprojections_filtered_as_vec: Vec<_> = reprojections.into_iter().enumerate().filter(|(i,_)| match_indices_filtered.contains(i)).map(|(_,c)| *c).collect();
        
        let reprojections_filtered = DVector::<Float>::from_vec(reprojections_filtered_as_vec);

        assert_eq!(matches_norm_filtered.len(), matches_filtered.len());
        assert_eq!(matches_norm_filtered.len(), landmarks_filtered.len());
        assert_eq!(matches_norm_filtered.len(), reprojections_filtered.nrows());

        match_norm_map.insert(cam_key, matches_norm_filtered);
        match_map.insert(cam_key, matches_filtered);
        landmark_map.insert(cam_key, landmarks_filtered);
        reprojection_error_map.insert(cam_key, reprojections_filtered);
    }

    for rejected_id in rejected_landmark_ids {
        first_landmark_sighting_map.remove(rejected_id);
    }
    assert!(!&first_landmark_sighting_map.is_empty());

}


pub fn compute_continuous_landmark_ids_from_unique_landmarks(
    unique_landmark_ids: &HashSet<usize>, rejected_landmark_ids_option: Option<&HashSet<usize>>) 
    -> (HashMap<usize,usize>, HashSet<usize>) {
    assert!(!unique_landmark_ids.is_empty());
    let mut free_ids = (0..unique_landmark_ids.len()).collect::<HashSet<usize>>();

    let mut old_new_map = HashMap::<usize,usize>::with_capacity(unique_landmark_ids.len());
    for old_id in unique_landmark_ids {
        assert!(!old_new_map.contains_key(&old_id));
        let free_id = free_ids.iter().next().unwrap().clone();
        free_ids.remove(&free_id);
        old_new_map.insert(*old_id, free_id);
    }
    
    assert!(free_ids.is_empty());

    let new_unique_landmark_ids = match (unique_landmark_ids,rejected_landmark_ids_option) {
        (unique_landmark_ids, Some(rejected_landmark_ids)) => unique_landmark_ids.iter().filter(|v| !rejected_landmark_ids.contains(v)).map(|v| *old_new_map.get(&v).expect("filter_by_rejected_landmark_ids: no id for unique_landmark_ids")).collect::<HashSet<_>>(),
        _ => (0..unique_landmark_ids.len()).collect::<HashSet<usize>>()
    };


    (old_new_map, new_unique_landmark_ids)

}

