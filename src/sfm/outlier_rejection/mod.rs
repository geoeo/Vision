extern crate nalgebra as na;

use na::{DVector,Matrix3x4,Matrix4xX, Vector4};
use crate::image::features::{matches::Match,Feature};
use crate::sensors::camera::Camera;
use crate::{float,Float};
use std::{collections::{HashMap,HashSet}};

pub mod dual;

pub fn calculate_reprojection_errors<Feat: Feature, C: Camera<Float>>(landmarks: &Matrix4xX<Float>, matches: &Vec<Match<Feat>>, transform_c1: &Matrix3x4<Float>, cam_1 :&C, transform_c2: &Matrix3x4<Float>, cam_2 :&C) -> DVector<Float> {
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

pub fn calcualte_disparities<Feat: Feature>(matches: &Vec<Match<Feat>>) -> DVector<Float> {
    let mut disparities = DVector::<Float>::zeros(matches.len());
    for i in 0..matches.len() {
        let m = &matches[i];
        disparities[i] = (m.get_feature_one().get_as_2d_point() - m.get_feature_two().get_as_2d_point()).norm()
    }
    disparities
}

pub fn reject_landmark_outliers<Feat: Feature + Clone>(
    landmark_map: &mut  HashMap<(usize, usize), Matrix4xX<Float>>, 
    reprojection_error_map: &mut HashMap<(usize, usize),DVector<Float>>, 
    match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
    match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
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

            let filtered_reprojection_errors_vec = reprojection_erros.iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| *v).collect::<Vec<Float>>();
            assert!(!&filtered_reprojection_errors_vec.is_empty());
            let filtered_reprojection_errors = DVector::<Float>::from_vec(filtered_reprojection_errors_vec);

            let filtered_landmarks_vec = landmarks.column_iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| v.into_owned()).collect::<Vec<Vector4<Float>>>();
            assert!(!&filtered_landmarks_vec.is_empty());
            let filtered_landmarks = Matrix4xX::<Float>::from_columns(&filtered_landmarks_vec);

            match_norm_map.insert(*key,filtered_matches_norm);
            match_map.insert(*key,filtered_matches);
            reprojection_error_map.insert(*key,filtered_reprojection_errors);
            landmark_map.insert(*key,filtered_landmarks);
        }

}

pub fn filter_by_rejected_landmark_ids<Feat: Feature + Clone>(
    rejected_landmark_ids: &HashSet<usize>,
    unique_landmark_ids: &mut HashSet<usize>,
    abs_landmark_map: &mut HashMap<usize,Matrix4xX<Float>>,
    match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>, 
    match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
    landmark_map: &mut  HashMap<(usize, usize), Matrix4xX<Float>>, 
    feature_map: &mut HashMap<usize, Vec<Feat>>,
    reprojection_error_map: &mut HashMap<(usize, usize),DVector<Float>>
) -> () {

    //TODO: update match map
    //TODO: update landmarks so that the position in landmark vec alignes with the matches
    //TODO: recomputes ids to be consecutive -> unique landmark ids 
    // for (cam_id, features) in feature_map {
    //     let accepted_enumerated_landmarks = features.drain(..).enumerate().filter(|(_,x)| !rejected_landmarks.contains(&x.get_landmark_id().expect("update_maps: no landmark id found"))).collect::<Vec<(usize,Feat)>>();
    //     assert!(features.is_empty());
    //     let (accepted_indices,accepted_landmarks): (Vec<usize>,Vec<Feat>) = accepted_enumerated_landmarks.into_iter().unzip();
    //     let accepted_landmaks = Matrix4xX::<Float>::from_element(accepted_indices.len(),1.0);

    //     features.extend(accepted_landmarks.into_iter())
    // }

}