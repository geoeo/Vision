extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use std::collections::HashMap;
use crate::image::{features::{Feature, Match, feature_track::FeatureTrack, solver_feature::SolverFeature},epipolar::tensor};
use crate::sensors::camera::Camera;

pub mod bundle_adjustment;
pub mod landmark; 

use na::{Vector3, Matrix3};
use crate::Float;


/**
 * We assume that the indices between paths and matches are consistent
 */
pub struct SFMConfig<C, C2, Feat: Feature> {
    root: usize,
    paths: Vec<Vec<usize>>,
    camera_map: HashMap<usize, C>,
    camera_map_ba: HashMap<usize, C2>,
    matches: Vec<Vec<Vec<Match<Feat>>>>,
    filtered_matches_by_tracks: Vec<Vec<Vec<Match<Feat>>>>,
    epipolar_alg: tensor::BifocalType,
    image_size: usize
}

impl<C: Camera<Float>, C2, Feat: Feature + Clone + std::cmp::PartialEq + SolverFeature> SFMConfig<C,C2, Feat> {

    //TODO: rework casting to be part of camera trait or super struct
    pub fn new(root: usize, paths: Vec<Vec<usize>>, camera_map: HashMap<usize, C>, camera_map_ba: HashMap<usize, C2>, matches: Vec<Vec<Vec<Match<Feat>>>>, epipolar_alg: tensor::BifocalType, image_size: usize) -> SFMConfig<C,C2,Feat> {
        for key in camera_map.keys() {
            assert!(camera_map_ba.contains_key(key));
        }

        for key in camera_map_ba.keys() {
            assert!(camera_map.contains_key(key));
        }

        let filtered_matches_by_tracks = Self::filter_by_max_tracks(&matches, image_size);

        SFMConfig{root, paths, camera_map, camera_map_ba, matches, filtered_matches_by_tracks, epipolar_alg, image_size}
    }


    pub fn root(&self) -> usize { self.root }
    pub fn paths(&self) -> &Vec<Vec<usize>> { &self.paths }
    pub fn camera_map(&self) -> &HashMap<usize, C> { &self.camera_map }
    pub fn camera_map_ba(&self) -> &HashMap<usize, C2> { &self.camera_map_ba }
    pub fn matches(&self) -> &Vec<Vec<Vec<Match<Feat>>>> { &self.matches }
    pub fn filtered_matches_by_tracks(&self) -> &Vec<Vec<Vec<Match<Feat>>>> { &self.filtered_matches_by_tracks }
    pub fn epipolar_alg(&self) -> tensor::BifocalType { self.epipolar_alg}
    pub fn image_size(&self) -> usize { self.image_size}

    pub fn compute_path_id_pairs(&self) -> Vec<Vec<(usize, usize)>> {
        let mut path_id_paris = Vec::<Vec::<(usize,usize)>>::with_capacity(self.paths.len());
        for sub_path in &self.paths {
            path_id_paris.push(
                sub_path.iter().enumerate().map(|(i,&id)| 
                    match i {
                        0 => (self.root,id),
                        idx => (sub_path[idx-1],id)
                    }
                ).collect()
            )
        }

        path_id_paris
    }

    pub fn compute_unqiue_ids_cameras_root_first(&self) -> (Vec<usize>, Vec<&C>) {
        let keys_sorted = self.get_sorted_camera_keys();
        let cameras_sorted = keys_sorted.iter().map(|id| self.camera_map.get(id).expect("compute_unqiue_ids_cameras_root_first: trying to get invalid camera")).collect::<Vec<&C>>();
        (keys_sorted,cameras_sorted)
    }

    pub fn compute_unqiue_ids_cameras_ba_root_first(&self) -> (Vec<usize>, Vec<&C2>) {
        let keys_sorted = self.get_sorted_camera_keys();
        let cameras_sorted = keys_sorted.iter().map(|id| self.camera_map_ba.get(id).expect("compute_unqiue_ids_cameras_lowp_root_first: trying to get invalid camera")).collect::<Vec<&C2>>();
        (keys_sorted,cameras_sorted)
    }

    fn get_sorted_camera_keys(&self) -> Vec<usize> {
        let number_of_keys = self.camera_map_ba.keys().len();
        let mut keys_sorted = Vec::<usize>::with_capacity(number_of_keys);
        // root has to first by design
        keys_sorted.push(self.root());
        keys_sorted.extend(self.paths.clone().into_iter().flatten().collect::<Vec<usize>>());
        keys_sorted.dedup();
        keys_sorted
    }

    fn filter_by_max_tracks(matches: &Vec<Vec<Vec<Match<Feat>>>>,  image_size: usize) -> Vec<Vec<Vec<Match<Feat>>>> {

        let mut filtered_matches = Vec::<Vec<Vec<Match<Feat>>>>::with_capacity(matches.len()); 
        let mut feature_tracks = Vec::<Vec<FeatureTrack<Feat>>>::with_capacity(matches.len());

        for i in 0..matches.len() {
            let path = &matches[i];
            let path_len = path.len();
            filtered_matches.push(Vec::<Vec<Match<Feat>>>::with_capacity(path_len));
            feature_tracks.push(Vec::<FeatureTrack<Feat>>::with_capacity(image_size));
            for img_idx in 0..path_len {
                filtered_matches[i].push(Vec::<Match<Feat>>::with_capacity(path[img_idx].len()));
            }
        }

        let max_path_len: usize = matches.iter().map(|x| x.len()).sum();
        
        //TODO: make this work feature tracks that happen on an image other than the previous
        for i in 0..matches.len() {
            let matches_for_path = matches[i].clone();
            let path_len = matches_for_path.len();
            for img_idx in 0..path_len {
                let current_matches = matches_for_path[img_idx].clone();
                for m in &current_matches {
                    match img_idx {
                        0 => feature_tracks[i].push(FeatureTrack::new(max_path_len,i, m)),
                        _ => {
                            let current_feature_one = &m.feature_one;
                            //TODO: Speed up with caching
                            for track in feature_tracks[i].iter_mut() {
                                if (track.get_current_feature() == current_feature_one.clone()) && 
                                   (track.get_path_img_id() == (i, img_idx-1)) {
                                    track.add(i,img_idx, m);
                                    break;
                                }
                            }
                        }
                    };
                }
            }
        }

        let max_track_lengths = feature_tracks.iter().map(|l| l.iter().map(|x| x.get_track_length()).reduce(|max, l| {
            if l > max { l } else { max }
        }).expect("filter_by_max_tracks: tracks is empty!")).collect::<Vec<usize>>();


        let max_tracks: Vec<Vec<FeatureTrack<Feat>>> = feature_tracks.into_iter().zip(max_track_lengths).map(| (xs, max) | xs.into_iter().filter(|x| x.get_track_length() == max).collect()).collect();

        for ts in &max_tracks {
            for t in ts {
                for (path_idx, img_idx, m) in t.get_track() {
                    (filtered_matches[*path_idx])[*img_idx].push(m.clone());
                }
            }
        }

        for path_idx in 0..matches.len() {
            let path = &matches[path_idx];
            for img_idx in 0..path.len() {
                (filtered_matches[path_idx])[img_idx].shrink_to_fit();
            }
        }

        filtered_matches
    }

    #[allow(non_snake_case)]
    pub fn compute_pairwise_cam_motions_with_filtered_matches(
            &self,
            perc_tresh: Float, 
            normalize_features: bool,
            epipolar_alg: tensor::BifocalType) 
        ->  (Vec<Vec<(usize,(Vector3<Float>,Matrix3<Float>))>>,Vec<Vec<Vec<Match<Feat>>>>) {
            let root_id = self.root();
            let root_cam = self.camera_map.get(&root_id).expect("compute_pairwise_cam_motions_for_path: could not get root cam");
            let mut all_states: Vec<Vec<(usize,(Vector3<Float>,Matrix3<Float>))>> = Vec::<Vec<(usize,(Vector3<Float>,Matrix3<Float>))>>::with_capacity(100);
            let mut all_filtered_matches: Vec<Vec<Vec<Match<Feat>>>> = Vec::<Vec<Vec<Match<Feat>>>>::with_capacity(100);
            for path_idx in 0..self.paths.len() {
                //TODO: investigate this cloning
                let path = self.paths[path_idx].clone();
                let matches = self.matches[path_idx].clone();
                let matches_tracks = self.filtered_matches_by_tracks[path_idx].clone();
                let mut states: Vec<(usize,(Vector3<Float>,Matrix3<Float>))> = Vec::<(usize,(Vector3<Float>,Matrix3<Float>))>::with_capacity(100);
                let mut filtered_matches: Vec<Vec<Match<Feat>>> = Vec::<Vec<Match<Feat>>>::with_capacity(100);
                for j in 0..matches_tracks.len() {
                    let tracks = &matches_tracks[j];
                    let all_matches = &matches[j];
                    let m = tracks;
                    let c1 = match j {
                        0 => root_cam,
                        idx => self.camera_map.get(&path[idx-1]).expect("compute_pairwise_cam_motions_for_path: could not get previous cam")
                    };
                    let id2 = path[j];
                    let c2 = self.camera_map.get(&id2).expect("compute_pairwise_cam_motions_for_path: could not get second camera");
                    let f0 = 1.0; // -> check this
                    let (e,f_m) = match epipolar_alg {
                        tensor::BifocalType::FUNDAMENTAL => {      
                            let f = tensor::fundamental::eight_point_hartley(m, false, f0); //TODO: make this configurable
                            
                            // let f_corr = tensor::fundamental::optimal_correction(&f, m, 1.0);
                            // let filtered = tensor::select_best_matches_from_fundamental(&f_corr,m,perc_tresh);
                            // (tensor::compute_essential(&f_corr,&c1.get_projection(),&c2.get_projection()), filtered)
            
                            let filtered = tensor::select_best_matches_from_fundamental(&f,m,perc_tresh);
                            //let filtered = tensor::filter_matches_from_fundamental(&f,m,3e0);


                            (tensor::compute_essential(&f,&c1.get_projection(),&c2.get_projection()), filtered)
                        },
                        tensor::BifocalType::ESSENTIAL => {
                            //TODO: put these in configs 
                            //Do NcR for
                            //let e = tensor::ransac_five_point_essential(m, c1, c2, 1e-2,1e5 as usize, 5);
                            let e = tensor::five_point_essential(m, c1, c2);
                            let f = tensor::compute_fundamental(&e, &c1.get_inverse_projection(), &c2.get_inverse_projection());
                            
                            //Seems to work better for olsen data 1e-1?
                            // let f_corr = tensor::fundamental::optimal_correction(&f, m, f0);
                            // let filtered =  tensor::select_best_matches_from_fundamental(&f_corr,m,perc_tresh);
                            // (tensor::compute_essential(&f_corr,&c1.get_projection(),&c2.get_projection()), filtered)
            
                            (e, tensor::select_best_matches_from_fundamental(&f,m,perc_tresh))
                        }
                    };
            
                    let (h,rotation,_) = tensor::decompose_essential_förstner(&e,&f_m,c1,c2);
                    let new_state = (id2,(h, rotation));
                    states.push(new_state);
                    filtered_matches.push(f_m);
                }

                all_states.push(states);
                all_filtered_matches.push(filtered_matches);
            }
        (all_states, all_filtered_matches)
    }

}