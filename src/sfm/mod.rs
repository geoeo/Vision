extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use std::collections::HashMap;
use crate::image::{features::{Feature, Match, feature_track::FeatureTrack},epipolar::tensor};
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

impl<C: Camera<Float>, C2, Feat: Feature + Clone> SFMConfig<C,C2, Feat> {

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

    //TODO: this is buggy
    fn filter_by_max_tracks(matches: &Vec<Vec<Vec<Match<Feat>>>>, image_size: usize) -> Vec<Vec<Vec<Match<Feat>>>> {

        let mut filtered_matches = Vec::<Vec<Vec<Match<Feat>>>>::with_capacity(matches.len()); 
        let mut feature_tracks = Vec::<FeatureTrack<Feat>>::with_capacity(image_size);

        for i in 0..matches.len() {
            let path = &matches[i];
            filtered_matches.push(Vec::<Vec<Match<Feat>>>::with_capacity(path.len()));
            for img_idx in 0..path.len() {
                filtered_matches[i].push(Vec::<Match<Feat>>::with_capacity(path[img_idx].len()));
            }
        }

        let max_path_len: usize = matches.iter().map(|x| x.len()).sum();
        
        //TODO: make this work feature tracks that happen on an image other than the first
        for i in 0..matches.len() {
            let path = &matches[i];
            let path_len = path.len();
            for img_idx in 0..path_len {
                let current_matches = &path[img_idx];
                for feat_idx in 0..current_matches.len() {
                    let m = &current_matches[feat_idx];
                    match img_idx {
                        0 => feature_tracks.push(FeatureTrack::new(max_path_len,i, m)),
                        _ => {
                            let current_feature_one = &m.feature_one;
                            //TODO: Speed up with caching
                            for track in feature_tracks.iter_mut() {
                                if (track.get_feature_current_id() == (current_feature_one.get_x_image(), current_feature_one.get_y_image())) && 
                                   (track.get_path_img_id() != (i, img_idx)) {
                                    track.add(i,img_idx, m);
                                }
                                
                            }
                        }
                    };
                }
                
            }
        }

        let max_track_length = feature_tracks.iter().map(|x| x.get_track_length()).reduce(|max, l| {
            if l > max { l } else { max }
        }).expect("filter_by_max_tracks: tracks is empty!");

        let max_tracks: Vec<FeatureTrack<Feat>> = feature_tracks.into_iter().filter(| x | x.get_track_length() == max_track_length).collect();

        //TODO: convert max tracks back into match vector
        for t in &max_tracks {
            for (path_idx, img_idx, m) in t.get_track() {
                (filtered_matches[*path_idx])[*img_idx].push(m.clone());
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
pub fn compute_pairwise_cam_motions_with_filtered_matches_for_path(
        &self,
        root_id: usize,
        camera_map: &HashMap<usize, C>,
        path: &Vec<usize>,
        matches: &Vec<Vec<Match<Feat>>>,
        matches_tracks: &Vec<Vec<Match<Feat>>>,
        epipolar_thresh: Float, 
        normalize_features: bool,
        epipolar_alg: tensor::BifocalType,
        decomp_alg: tensor::EssentialDecomposition) 
    ->  (Vec<(usize,(Vector3<Float>,Matrix3<Float>))>,Vec<Vec<Match<Feat>>>) {
    let root_cam = camera_map.get(&root_id).expect("compute_pairwise_cam_motions_for_path: could not get root cam");
    matches.iter().zip(matches_tracks.iter()).enumerate().map(|(i,(m,tracks))| {
        let c1 = match i {
            0 => root_cam,
            idx => camera_map.get(&path[idx-1]).expect("compute_pairwise_cam_motions_for_path: could not get previous cam")
        };
        let id2 = path[i];
        let c2 = camera_map.get(&id2).expect("compute_pairwise_cam_motions_for_path: could not get second camera");
        let f0 = 1.0; // -> check this
        let (e,f_m) = match epipolar_alg {
            tensor::BifocalType::FUNDAMENTAL => {      
                let f = tensor::fundamental::eight_point_hartley(tracks, false, f0); //TODO: make this configurable
                
                // let f_corr = tensor::fundamental::optimal_correction(&f, m, 1.0);
                // let filtered = tensor::filter_matches_from_fundamental(&f_corr,m,epipolar_thresh, c1,c2);
                // (tensor::compute_essential(&f_corr,&c1.get_projection(),&c2.get_projection()), filtered)

                let filtered = tensor::filter_matches_from_fundamental(&f,tracks,epipolar_thresh, c1,c2);
                (tensor::compute_essential(&f,&c1.get_projection(),&c2.get_projection()), filtered)
            },
            tensor::BifocalType::ESSENTIAL => {
                //TODO: put these in configs 
                //Do NcR for
                //let e = tensor::ransac_five_point_essential(m, c1, c2, 1e-2,1e5 as usize, 5);
                let e = tensor::five_point_essential(tracks, c1, c2);
                let f = tensor::compute_fundamental(&e, &c1.get_inverse_projection(), &c2.get_inverse_projection());
                
                //Seems to work better for olsen data 1e-1?
                // let f_corr = tensor::fundamental::optimal_correction(&f, m, f0);
                // let filtered =  tensor::filter_matches_from_fundamental(&f_corr,m,epipolar_thresh,c1,c2);
                // (tensor::compute_essential(&f_corr,&c1.get_projection(),&c2.get_projection()), filtered)

                (e, tensor::filter_matches_from_fundamental(&f,tracks,epipolar_thresh,c1,c2))
            }
        };

        let (h,rotation,_) = match decomp_alg {
            tensor::EssentialDecomposition::FÖRSNTER => tensor::decompose_essential_förstner(&e,&f_m,c1,c2),
            tensor::EssentialDecomposition::KANATANI => tensor::decompose_essential_kanatani(&e,&f_m, false)
        };
        let new_state = (id2,(h, rotation));
        (new_state, m.clone())
    }).unzip()
}


#[allow(non_snake_case)]
pub fn compute_pairwise_cam_motions_with_filtered_matches(
        &self,
        epipolar_thresh: Float, 
        normalize_features: bool,
        epipolar_alg: tensor::BifocalType,
        decomp_alg: tensor::EssentialDecomposition) 
    ->  (Vec<Vec<(usize,(Vector3<Float>,Matrix3<Float>))>>,Vec<Vec<Vec<Match<Feat>>>>) {
        let root_id = self.root();
        let camera_map = self.camera_map();
        (0..self.paths.len()).map(|path_idx| {
            //TODO: investigate this cloning
            let path = self.paths[path_idx].clone();
            let matches = self.matches[path_idx].clone();
            let matches_tracks = self.filtered_matches_by_tracks[path_idx].clone();
            self.compute_pairwise_cam_motions_with_filtered_matches_for_path(
                root_id,
                camera_map,
                &path,
                &matches,
                &matches_tracks,
                epipolar_thresh,
                normalize_features,
                epipolar_alg, 
                decomp_alg)
            }
    ).unzip()
    
}

}