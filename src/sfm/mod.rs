extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

pub mod bundle_adjustment;
pub mod landmark;
pub mod epipolar;
pub mod triangulation;
pub mod quest;
pub mod rotation_avg;

use std::collections::HashMap;
use crate::image::{features::{Feature, Match, feature_track::FeatureTrack, solver_feature::SolverFeature, subsample_matches}};
use crate::sfm::{epipolar::tensor, triangulation::Triangulation, rotation_avg::{optimize_rotations_with_rcd_per_track,optimize_rotations_with_rcd}};
use crate::sensors::camera::Camera;
use crate::numerics::{lie::angular_distance, pose::{to_parts,from_matrix,se3}};

use na::{Vector3, Matrix3, Isometry3};
use crate::Float;


/**
 * We assume that the indices between paths and matches are consistent
 */
pub struct SFMConfig<C, C2, Feat: Feature> {
    root: usize,
    paths: Vec<Vec<usize>>,
    camera_map: HashMap<usize, C>,
    camera_map_ba: HashMap<usize, C2>, //TODO: unfiy camera map
    match_map: HashMap<(usize, usize), Vec<Match<Feat>>>,
    pose_map: HashMap<(usize, usize), Isometry3<Float>>,
    epipolar_alg: tensor::BifocalType,
    triangulation: Triangulation
}

impl<C: Camera<Float>, C2, Feat: Feature + Clone + std::cmp::PartialEq + SolverFeature> SFMConfig<C,C2, Feat> {

    //TODO: rework casting to be part of camera trait or super struct 
    pub fn new(root: usize, paths: Vec<Vec<usize>>, 
        camera_map: HashMap<usize, C>, 
        camera_map_ba: HashMap<usize, C2>, matches: Vec<Vec<Vec<Match<Feat>>>>, 
        epipolar_alg: tensor::BifocalType, 
        triangulation: Triangulation, 
        filter_tracks: bool,
        perc_tresh: Float, 
        epipolar_thresh: Float, 
        angular_thresh: Float,
        refine_rotation_via_rcd: bool,
        image_width: usize,
        image_height: usize) -> SFMConfig<C,C2,Feat> {
        for key in camera_map.keys() {
            assert!(camera_map_ba.contains_key(key));
        }

        for key in camera_map_ba.keys() {
            assert!(camera_map.contains_key(key));
        }

        let image_size = image_width*image_height;
        // Filteres matches according to feature consitency along a path.
        let accepted_matches = match filter_tracks {
            true => Self::filter_by_max_tracks(&matches, image_size),
            false => matches
        };

        let (mut pose_map, match_map) = Self::compute_pose_and_feature_maps(
            root,
            &paths,
            &camera_map,
            &accepted_matches,
            perc_tresh, 
            epipolar_thresh,
            epipolar_alg,
            image_width,
            image_height
        );

        if refine_rotation_via_rcd {
            Self::refine_rotation_by_rcd(root, &paths, &mut pose_map, angular_thresh,);
        }

        SFMConfig{root, paths, camera_map, camera_map_ba, match_map, pose_map, epipolar_alg, triangulation}
    }


    pub fn root(&self) -> usize { self.root }
    pub fn paths(&self) -> &Vec<Vec<usize>> { &self.paths }
    pub fn camera_map(&self) -> &HashMap<usize, C> { &self.camera_map }
    pub fn camera_map_ba(&self) -> &HashMap<usize, C2> { &self.camera_map_ba }
    pub fn epipolar_alg(&self) -> tensor::BifocalType { self.epipolar_alg}
    pub fn triangulation(&self) -> Triangulation { self.triangulation}
    pub fn match_map(&self) -> &HashMap<(usize, usize), Vec<Match<Feat>>> {&self.match_map}
    pub fn pose_map(&self) -> &HashMap<(usize, usize), Isometry3<Float>> {&self.pose_map}

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

    fn filter_by_max_tracks(all_paths: &Vec<Vec<Vec<Match<Feat>>>>,  image_size: usize) -> Vec<Vec<Vec<Match<Feat>>>> {

        let mut filtered_matches = Vec::<Vec<Vec<Match<Feat>>>>::with_capacity(all_paths.len()); 
        let mut feature_tracks = Vec::<Vec<FeatureTrack<Feat>>>::with_capacity(all_paths.len());

        for path_idx in 0..all_paths.len() {
            let path = &all_paths[path_idx];
            let path_len = path.len();
            filtered_matches.push(Vec::<Vec<Match<Feat>>>::with_capacity(path_len));
            feature_tracks.push(Vec::<FeatureTrack<Feat>>::with_capacity(image_size));
            for img_idx in 0..path_len {
                filtered_matches[path_idx].push(Vec::<Match<Feat>>::with_capacity(path[img_idx].len()));
            }
        }

        let max_path_len: usize = all_paths.iter().map(|x| x.len()).sum();
        for path_idx in 0..all_paths.len() {
            let matches_for_path = all_paths[path_idx].clone();
            let path_len = matches_for_path.len();
            for img_idx in 0..path_len {
                let current_matches = matches_for_path[img_idx].clone();
                for m in &current_matches {
                    match img_idx {
                        0 => feature_tracks[path_idx].push(FeatureTrack::new(max_path_len, path_idx, m)),
                        _ => {
                                let current_feature_one = &m.feature_one;
                                let mut found_track = false;
                                //TODO: Speed up with caching
                                for track in feature_tracks[path_idx].iter_mut() {
                                    if (track.get_current_feature() == current_feature_one.clone()) && 
                                        (track.get_path_img_id() == (path_idx, img_idx-1)) {
                                        track.add(path_idx,img_idx, m);
                                        found_track = true;
                                        break;
                                    }
                                }
                                if !found_track {
                                    feature_tracks[path_idx].push(FeatureTrack::new(max_path_len, path_idx, m));
                                }
                        }
                    };
                }
            }
        }

        let max_track_lengths = feature_tracks.iter().map(|l| l.iter().map(|x| x.get_track_length()).reduce(|max, l| {
            if l > max { l } else { max }
        }).expect("filter_by_max_tracks: tracks is empty!")).collect::<Vec<usize>>();

        println!("Max Track len: {:?}", max_track_lengths);

        let max_tracks: Vec<Vec<FeatureTrack<Feat>>> = feature_tracks.into_iter().zip(max_track_lengths).map(| (xs, max) | xs.into_iter().filter(|x| x.get_track_length() == max).collect()).collect();

        for ts in &max_tracks {
            for t in ts {
                for (path_idx, img_idx, m) in t.get_track() {
                    (filtered_matches[*path_idx])[*img_idx].push(m.clone());
                }
            }
        }

        for path_idx in 0..all_paths.len() {
            let path = &all_paths[path_idx];
            for img_idx in 0..path.len() {
                (filtered_matches[path_idx])[img_idx].shrink_to_fit();
            }
        }

        filtered_matches
    } 

    pub fn compute_path_pairs_as_list(&self) -> Vec<Vec<(usize,usize)>> {
        let number_of_paths = self.paths.len();
        let mut all_path_pairs = Vec::<Vec<(usize,usize)>>::with_capacity(number_of_paths);
        for path_idx in 0..number_of_paths {
            let path = &self.paths[path_idx];
            let mut path_pair = Vec::<(usize,usize)>::with_capacity(path.len());
            for j in 0..path.len() {
                let id1 = match j {
                    0 => self.root(),
                    idx => path[idx-1]
                };
                let id2 = path[j];
                path_pair.push((id1,id2));
            }
            all_path_pairs.push(path_pair);
        }
        all_path_pairs
    }

    pub fn compute_lists_from_maps(&self)->  (Vec<Vec<((usize,usize),(Vector3<Float>, Matrix3<Float>))>>,Vec<Vec<Vec<Match<Feat>>>>){
        let number_of_paths = self.paths.len();
        let mut all_states = Vec::<Vec<((usize,usize),(Vector3<Float>,Matrix3<Float>))>>::with_capacity(number_of_paths);
        let mut all_filtered_matches = Vec::<Vec<Vec<Match<Feat>>>>::with_capacity(number_of_paths);
        for path_idx in 0..number_of_paths {
            let path = self.paths[path_idx].clone();
            let mut states = Vec::<((usize,usize),(Vector3<Float>,Matrix3<Float>))>::with_capacity(path.len());
            let mut filtered_matches = Vec::<Vec<Match<Feat>>>::with_capacity(path.len());

            for j in 0..path.len() {
                let id1 = match j {
                    0 => self.root(),
                    idx => path[idx-1]
                };
                let id2 = path[j];
                let key = (id1,id2);
                let isometry = match self.pose_map.get(&key) {
                    Some(iso) => iso.clone(),
                    None => Isometry3::identity()
                };
                states.push((key,to_parts(&isometry)));
                filtered_matches.push(self.match_map.get(&key).expect("invalid key for matches in SFM config").clone());
            }
            all_states.push(states);
            all_filtered_matches.push(filtered_matches);

        }
        (all_states, all_filtered_matches)
    }

    #[allow(non_snake_case)]
    fn compute_pose_and_feature_maps(
            root: usize,
            paths: &Vec<Vec<usize>>,
            camera_map: &HashMap<usize, C>,
            matches: &Vec<Vec<Vec<Match<Feat>>>>,
            perc_tresh: Float, 
            epipolar_tresh: Float,
            epipolar_alg: tensor::BifocalType,
            image_width: usize,
            image_height: usize) 
        ->  (HashMap<(usize, usize), Isometry3<Float>>, HashMap<(usize, usize), Vec<Match<Feat>>>) {
            let number_of_paths = paths.len();
            let map_capacity = 100*number_of_paths;
            let mut pose_map = HashMap::<(usize, usize), Isometry3<Float>>::with_capacity(map_capacity);
            let mut match_map = HashMap::<(usize, usize), Vec<Match<Feat>>>::with_capacity(map_capacity);
            for path_idx in 0..number_of_paths {
                let path = paths[path_idx].clone();
                let matche_per_path = matches[path_idx].clone();
                for j in 0..path.len() {
                    let id1 = match j {
                        0 => root,
                        idx => path[idx-1]
                    };
                    let id2 = path[j];
                    let c1 = camera_map.get(&id1).expect("compute_pairwise_cam_motions_for_path: could not get previous cam");
                    let c2 = camera_map.get(&id2).expect("compute_pairwise_cam_motions_for_path: could not get second camera");
                    let m = &matche_per_path[j];
                    let f0 = 1.0; // -> check this
                    let (e,f_m) = match epipolar_alg {
                        tensor::BifocalType::FUNDAMENTAL => {      
                            let f = tensor::fundamental::eight_point_hartley(m, false, f0); //TODO: make this configurable
                            
                            // let f_corr = tensor::fundamental::optimal_correction(&f, m, 1.0);
                            // let filtered = tensor::select_best_matches_from_fundamental(&f,m,perc_tresh, epipolar_tresh);
                            // (tensor::compute_essential(&f_corr,&c1.get_projection(),&c2.get_projection()), filtered)
            
                            let filtered = tensor::select_best_matches_from_fundamental(&f,m,perc_tresh, epipolar_tresh);
                            (tensor::compute_essential(&f,&c1.get_projection(),&c2.get_projection()), filtered)
                        },
                        tensor::BifocalType::ESSENTIAL => {
                            let e = tensor::five_point_essential(m, c1, c2);
                            let f = tensor::compute_fundamental(&e, &c1.get_inverse_projection(), &c2.get_inverse_projection());
                            (e, tensor::select_best_matches_from_fundamental(&f,m,perc_tresh,epipolar_tresh))
                        },
                        tensor::BifocalType::QUEST => {
                            let e = quest::quest_ransac(m, c1, c2, 1e-2,1e4 as usize);
                            let f = tensor::compute_fundamental(&e, &c1.get_inverse_projection(), &c2.get_inverse_projection());
                            (e, tensor::select_best_matches_from_fundamental(&f,m,perc_tresh,epipolar_tresh))
                        }
                    };
                    
                    //TODO
                    let f_m_subsampled = subsample_matches(f_m,image_width,image_height);
                    
                    // The pose transforms id2 into the coordiante system of id1
                    let (h,rotation,_) = tensor::decompose_essential_förstner(&e,&f_m_subsampled,c1,c2);
                    let se3 = se3(&h,&rotation);
                    let isometry = from_matrix(&se3);
                    let some_pose_old_val = pose_map.insert((id1, id2), isometry);
                    let some_match_old_val = match_map.insert((id1, id2), f_m_subsampled);
                    assert!(some_pose_old_val.is_none());
                    assert!(some_match_old_val.is_none());

                }
            }
        (pose_map, match_map)
    }

    fn refine_rotation_by_rcd(root: usize, paths: &Vec<Vec<usize>>, pose_map: &mut HashMap<(usize, usize), Isometry3<Float>>, angular_thresh: Float,) -> () {
        let number_of_paths = paths.len(); 
        let mut initial_cam_motions_per_path = Vec::<Vec<((usize, usize), Matrix3<Float>)>>::with_capacity(number_of_paths);
        for path_idx in 0..number_of_paths {
            let path = &paths[path_idx];
            let path_length = path.len();
            let mut cam_motions =  Vec::<((usize, usize), Matrix3<Float>)>::with_capacity(path_length);

            for j in 0..path_length {
                let id1 = match j {
                    0 => root,
                    idx => path[idx-1]
                };
                let id2 = path[j];
                let key = (id1,id2);
                let initial_pose = pose_map.get(&key).expect("No pose found for rcd rotation in pose map");
                cam_motions.push((key,initial_pose.rotation.to_rotation_matrix().matrix().to_owned()));
            }
            initial_cam_motions_per_path.push(cam_motions);
        }
        let initial_cam_rotations_per_path_rcd = optimize_rotations_with_rcd_per_track(&initial_cam_motions_per_path);
        for i in 0..initial_cam_rotations_per_path_rcd.len(){
            let path_len = initial_cam_rotations_per_path_rcd[i].len();
            for j in 0..path_len {
                let (key,rcd_rot) = initial_cam_rotations_per_path_rcd[i][j];
                let initial_pose = pose_map.get(&key).expect("No pose found for rcd rotation in pose map");
                let initial_rot = initial_pose.rotation.to_rotation_matrix().matrix().to_owned();
                let angular_distance_initial = angular_distance(&initial_rot);
                let angular_distance_rcd = angular_distance(&rcd_rot);

                println!("initial r : {}", initial_rot);
                println!("rcd r : {}",rcd_rot);
                println!("initial ang dist : {}", angular_distance_initial);
                println!("rcd ang dist : {}", angular_distance_rcd);

                //TODO: investigate this
                if (angular_distance_rcd-angular_distance_initial).abs() <= angular_thresh {
                    let new_se3 = se3(&initial_pose.translation.vector,&rcd_rot);
                    let old_pose = pose_map.insert(key, from_matrix(&new_se3));
                    assert!(old_pose.is_some());
                }
            }
        }

    }

}