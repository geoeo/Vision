extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

pub mod bundle_adjustment;
pub mod landmark;
pub mod epipolar;
pub mod triangulation;
pub mod quest;
pub mod rotation_avg;

use std::collections::{HashMap,HashSet};
use crate::image::{features::{Feature, Match, feature_track::FeatureTrack, solver_feature::SolverFeature}};
use crate::sfm::{epipolar::tensor, epipolar::compute_linear_normalization,
    triangulation::{Triangulation, triangulate_matches}, 
    rotation_avg::{optimize_rotations_with_rcd_per_track,optimize_rotations_with_rcd}};
use crate::sensors::camera::Camera;
use crate::numerics::{pose::{to_parts,from_matrix,se3}};

use na::{DVector, Matrix4xX, Vector3, Vector4, Matrix3, Isometry3};
use crate::{float,Float};


/**
 * We assume that the indices between paths and matches are consistent
 */
pub struct SFMConfig<C, C2, Feat: Feature> {
    root: usize,
    paths: Vec<Vec<usize>>,
    camera_map_highp: HashMap<usize, C>,
    camera_map_lowp: HashMap<usize, C2>,
    match_map: HashMap<(usize, usize), Vec<Match<Feat>>>,
    pose_map: HashMap<(usize, usize), Isometry3<Float>>, // The pose transforms tuple id 2 into the coordiante system of tuple id 1
    landmark_map: HashMap<(usize, usize), Matrix4xX<Float>>,
    reprojection_error_map: HashMap<(usize, usize),DVector<Float>>,
    epipolar_alg: tensor::BifocalType,
    triangulation: Triangulation
}


pub fn compute_path_pairs_as_list(root: usize, paths: &Vec<Vec<usize>>) -> Vec<Vec<(usize,usize)>> {
    let number_of_paths = paths.len();
    let mut all_path_pairs = Vec::<Vec<(usize,usize)>>::with_capacity(number_of_paths);
    for path_idx in 0..number_of_paths {
        let path = &paths[path_idx];
        let mut path_pair = Vec::<(usize,usize)>::with_capacity(path.len());
        for j in 0..path.len() {
            let id1 = match j {
                0 => root,
                idx => path[idx-1]
            };
            let id2 = path[j];
            path_pair.push((id1,id2));
        }
        all_path_pairs.push(path_pair);
    }
    all_path_pairs
}

impl<C: Camera<Float>, C2, Feat: Feature + Clone + std::cmp::PartialEq + SolverFeature> SFMConfig<C,C2, Feat> {

    //TODO: rework casting to be part of camera trait or super struct 
    pub fn new(root: usize, paths: Vec<Vec<usize>>, 
        camera_map: HashMap<usize, C>, 
        camera_map_ba: HashMap<usize, C2>, matches: Vec<Vec<Vec<Match<Feat>>>>, 
        epipolar_alg: tensor::BifocalType, 
        triangulation: Triangulation, 
        perc_tresh: Float, 
        epipolar_thresh: Float, 
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
        let accepted_matches = Self::filter_by_max_tracks(&matches, image_size);

        let _ = Self::check_for_duplicate_pixel_entries(&accepted_matches);

        let match_map_initial = Self::generate_match_map(root, &paths,accepted_matches);

        let (mut pose_map,mut match_map) = Self::compute_pose_map(
            root,
            &paths,
            &camera_map,
            match_map_initial,
            perc_tresh, 
            epipolar_thresh,
            epipolar_alg
        );

        let (mut landmark_map, mut reprojection_error_map, min_reprojection_error_initial, max_reprojection_error_initial) =  Self::compute_landmarks_and_reprojection_maps(root,&paths,&pose_map,&match_map,&camera_map,triangulation);
        println!("SFM Config Max Reprojection Error 1): {}, Min Reprojection Error: {}", max_reprojection_error_initial, min_reprojection_error_initial);
        let mut landmark_cutoff = Self::calc_landmark_cutoff(max_reprojection_error_initial);    
        Self::reject_landmark_outliers( &mut landmark_map, &mut reprojection_error_map, &mut match_map, landmark_cutoff);
        if refine_rotation_via_rcd {
            let new_pose_map = Self::refine_rotation_by_rcd(root, &paths, &pose_map);
            let (new_landmark_map, new_reprojection_error_map, _, _) =  Self::compute_landmarks_and_reprojection_maps(root,&paths,&new_pose_map,&match_map,&camera_map,triangulation);
            let keys = landmark_map.keys().map(|k| *k).collect::<Vec<_>>();
            for key in keys {
                let new_reprojection_errors = new_reprojection_error_map.get(&key).unwrap();
                let current_reprojection_errors = reprojection_error_map.get_mut(&key).unwrap();

                if new_reprojection_errors.mean() < current_reprojection_errors.mean(){
                    landmark_map.insert(key,new_landmark_map.get(&key).unwrap().clone());
                    reprojection_error_map.insert(key,new_reprojection_error_map.get(&key).unwrap().clone());
                    pose_map.insert(key,new_pose_map.get(&key).unwrap().clone());
                }
            }
        }

        let (min_reprojection_error_refined, max_reprojection_error_refined) = Self::compute_reprojection_ranges(&reprojection_error_map);
        println!("SFM Config Max Reprojection Error 2): {}, Min Reprojection Error: {}", max_reprojection_error_refined, min_reprojection_error_refined);
        landmark_cutoff = Self::calc_landmark_cutoff(max_reprojection_error_refined);    
        Self::reject_landmark_outliers(&mut landmark_map, &mut reprojection_error_map, &mut match_map, landmark_cutoff);

        Self::recompute_landmark_ids(&mut match_map);

        SFMConfig{root, paths, camera_map_highp: camera_map, camera_map_lowp: camera_map_ba, match_map, pose_map, epipolar_alg, landmark_map, reprojection_error_map,triangulation}
    }


    pub fn root(&self) -> usize { self.root }
    pub fn paths(&self) -> &Vec<Vec<usize>> { &self.paths }
    pub fn camera_map_highp(&self) -> &HashMap<usize, C> { &self.camera_map_highp }
    pub fn camera_map_lowp(&self) -> &HashMap<usize, C2> { &self.camera_map_lowp }
    pub fn epipolar_alg(&self) -> tensor::BifocalType { self.epipolar_alg}
    pub fn triangulation(&self) -> Triangulation { self.triangulation}
    pub fn match_map(&self) -> &HashMap<(usize, usize), Vec<Match<Feat>>> {&self.match_map}
    pub fn pose_map(&self) -> &HashMap<(usize, usize), Isometry3<Float>> {&self.pose_map}
    pub fn landmark_map(&self) -> &HashMap<(usize, usize), Matrix4xX<Float>> {&self.landmark_map}
    pub fn reprojection_error_map(&self) -> &HashMap<(usize, usize), DVector<Float>> {&self.reprojection_error_map}

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
        let cameras_sorted = keys_sorted.iter().map(|id| self.camera_map_highp.get(id).expect("compute_unqiue_ids_cameras_root_first: trying to get invalid camera")).collect::<Vec<&C>>();
        (keys_sorted,cameras_sorted)
    }

    pub fn compute_unqiue_ids_cameras_ba_root_first(&self) -> (Vec<usize>, Vec<&C2>) {
        let keys_sorted = self.get_sorted_camera_keys();
        let cameras_sorted = keys_sorted.iter().map(|id| self.camera_map_lowp.get(id).expect("compute_unqiue_ids_cameras_lowp_root_first: trying to get invalid camera")).collect::<Vec<&C2>>();
        (keys_sorted,cameras_sorted)
    }

    fn check_for_duplicate_pixel_entries(matches: &Vec<Vec<Vec<Match<Feat>>>>) -> bool {
        let mut duplicates_found = false;
        for path in matches{
            for tracks in path {
                let mut pixel_map = HashMap::<(usize,usize), (Float, Float,Float, Float)>::with_capacity(1000*1000);
                for m in tracks {
                    let f = m.feature_one.get_as_2d_point();
                    let f2 = m.feature_two.get_as_2d_point();
                    let old_value = pixel_map.insert((f[0].trunc() as usize, f[1].trunc() as usize), (f[0], f[1], f2[0], f2[1]));
                    if old_value.is_some() {
                        duplicates_found = true;
                        println!("Warning: duplicate source entries in track: new: (x: {} y: {}, x f: {} y f: {}), old: (x: {} y: {}, x f: {} y f: {})",  f[0], f[1], f2[0], f2[1], old_value.unwrap().0, old_value.unwrap().1,old_value.unwrap().2, old_value.unwrap().3);
                    }
                }
            }
        }
        duplicates_found
    }

    //TODO: Doesnt really work robustly
    fn calc_landmark_cutoff(max_reprojection_error: Float) -> Float {
        //max_reprojection_error
        0.95*max_reprojection_error
    }

    fn generate_match_map(root: usize, paths: &Vec<Vec<usize>>, matches: Vec<Vec<Vec<Match<Feat>>>>) -> HashMap<(usize,usize), Vec<Match<Feat>>> {
        let number_of_paths = paths.len();
        let map_capacity = calc_map_capacity(number_of_paths);
        let mut match_map = HashMap::<(usize, usize), Vec<Match<Feat>>>::with_capacity(map_capacity);
        for path_idx in 0..number_of_paths {
            let path = paths[path_idx].clone();
            let match_per_path = matches[path_idx].clone();
            for j in 0..path.len() {
                let id1 = match j {
                    0 => root,
                    idx => path[idx-1]
                };
                let id2 = path[j];
                let m = &match_per_path[j];
                match_map.insert((id1, id2), m.clone());
            }
        }

        match_map
    }

    fn reject_landmark_outliers(
        landmark_map: &mut  HashMap<(usize, usize), Matrix4xX<Float>>, 
        reprojection_error_map: &mut HashMap<(usize, usize),DVector<Float>>, 
        match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        landmark_cutoff: Float){
            let keys = match_map.keys().map(|key| *key).collect::<Vec<_>>();
            for key in &keys {
                let reprojection_erros = reprojection_error_map.get(key).unwrap();
                let matches = match_map.get(key).unwrap();
                let landmarks = landmark_map.get(key).unwrap();

                let accepted_indices = reprojection_erros.iter().enumerate().filter(|&(_,v)| *v < landmark_cutoff).map(|(idx,_)| idx).collect::<HashSet<usize>>();
                let filtered_matches = matches.iter().enumerate().filter(|(idx,_)|accepted_indices.contains(idx)).map(|(_,v)| v.clone()).collect::<Vec<_>>();
                assert!(!&filtered_matches.is_empty());

                let filtered_reprojection_errors_vec = reprojection_erros.iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| *v).collect::<Vec<Float>>();
                assert!(!&filtered_reprojection_errors_vec.is_empty());
                let filtered_reprojection_errors = DVector::<Float>::from_vec(filtered_reprojection_errors_vec);

                let filtered_landmarks_vec = landmarks.column_iter().enumerate().filter(|(idx,_)| accepted_indices.contains(idx)).map(|(_,v)| v.into_owned()).collect::<Vec<Vector4<Float>>>();
                assert!(!&filtered_landmarks_vec.is_empty());
                let filtered_landmarks = Matrix4xX::<Float>::from_columns(&filtered_landmarks_vec);

                match_map.insert(*key,filtered_matches);
                reprojection_error_map.insert(*key,filtered_reprojection_errors);
                landmark_map.insert(*key,filtered_landmarks);
            }

    }

    fn compute_landmarks_and_reprojection_maps(root: usize, paths: &Vec<Vec<usize>>, 
        pose_map: &HashMap<(usize, usize), Isometry3<Float>>, 
        match_map: &HashMap<(usize, usize), 
        Vec<Match<Feat>>>, camera_map: &HashMap<usize, C>,
        triangulation: Triangulation) -> (HashMap<(usize,usize),Matrix4xX<Float>>,HashMap<(usize,usize), DVector<Float>>, Float, Float) {

        let mut triangulated_match_map = HashMap::<(usize,usize),Matrix4xX<Float>>::with_capacity(match_map.len());
        let mut reprojection_map = HashMap::<(usize,usize),DVector<Float>>::with_capacity(match_map.len());
        let path_pairs = compute_path_pairs_as_list(root,paths);
        let mut max_reprojection_error = float::MIN;
        let mut min_reprojection_error = float::MAX;

        for path in &path_pairs{
            for path_pair in path {
                let (trigulated_matches,reprojection_errors) = triangulate_matches(*path_pair,&pose_map,&match_map,&camera_map,triangulation);
                max_reprojection_error = max_reprojection_error.max(reprojection_errors.max()); 
                min_reprojection_error = min_reprojection_error.min(reprojection_errors.min());
                triangulated_match_map.insert(*path_pair,trigulated_matches);
                reprojection_map.insert(*path_pair,reprojection_errors);
            }
        }

        (triangulated_match_map, reprojection_map, min_reprojection_error, max_reprojection_error)
    }

    fn compute_reprojection_ranges(reprojection_map: &HashMap<(usize,usize), DVector<Float>>) -> (Float, Float) {
        let mut max_reprojection_error = float::MIN;
        let mut min_reprojection_error = float::MAX;

        for (_,reprojection_errors) in reprojection_map.iter(){
            max_reprojection_error = max_reprojection_error.max(reprojection_errors.max()); 
            min_reprojection_error = min_reprojection_error.min(reprojection_errors.min());   
        }

        (min_reprojection_error, max_reprojection_error)
    }

    fn get_sorted_camera_keys(&self) -> Vec<usize> {
        let number_of_keys = self.camera_map_lowp.keys().len();
        let mut keys_sorted = Vec::<usize>::with_capacity(number_of_keys);
        // root has to first by design
        keys_sorted.push(self.root());
        keys_sorted.extend(self.paths.clone().into_iter().flatten().collect::<Vec<usize>>());
        keys_sorted.dedup();
        keys_sorted
    }

    //TODO: merges in tracks originating at the root
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

        let mut landmark_id = 0;
        let max_path_len: usize = all_paths.iter().map(|x| x.len()).sum();
        for path_idx in 0..all_paths.len() {
            let matches_for_path = all_paths[path_idx].clone();
            let path_len = matches_for_path.len();
            for img_idx in 0..path_len {
                let current_matches = matches_for_path[img_idx].clone();
                let mut pixel_set = HashSet::<(usize,usize)>::with_capacity(current_matches.len());
                for m in &current_matches {
                    let f1_x = m.feature_one.get_x_image();
                    let f1_y = m.feature_one.get_y_image();
                    let k = (f1_x,f1_y);
                    match img_idx {
                        0 => {
                            if !pixel_set.contains(&k) {
                                let mut id : Option<usize> = None;
                                for j in 0..feature_tracks.len() {
                                    for i in 0..feature_tracks[j].len() {
                                        let track = &feature_tracks[j][i];
                                        if track.get_first_feature_start() == m.feature_one {
                                            id = Some(track.get_track_id());
                                            break;
                                        } 
                                    }
                                    if id.is_some() {
                                        break;
                                    }
                                }
                                
                                if id.is_some() {
                                    feature_tracks[path_idx].push(FeatureTrack::new(max_path_len, path_idx, 0, id.unwrap(), m));
                                } else {
                                    feature_tracks[path_idx].push(FeatureTrack::new(max_path_len, path_idx, 0, landmark_id, m));
                                    landmark_id +=1;

                                }
                                pixel_set.insert(k);
                            }
                        },
                        img_idx => {
                            let current_feature_one = &m.feature_one;
                            let mut found_track = false;
                            //TODO: Speed up with caching
                            for track in feature_tracks[path_idx].iter_mut() {
                                if (track.get_current_feature_dest() == current_feature_one.clone()) && 
                                    (track.get_path_img_id() == (path_idx, img_idx-1)) &&
                                    !pixel_set.contains(&k) {
                                    track.add(path_idx,img_idx, m);
                                    found_track = true;
                                    pixel_set.insert(k);
                                    break;
                                }
                            }
                            if !(found_track || pixel_set.contains(&k)){
                                feature_tracks[path_idx].push(FeatureTrack::new(max_path_len, path_idx, img_idx, landmark_id, m));
                                landmark_id +=1;
                                pixel_set.insert(k);
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

    fn recompute_landmark_ids(match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>) -> () {
        let mut old_max_val = 0;

        let mut existing_ids = HashSet::<usize>::with_capacity(100000);
        for (_,val) in match_map.iter() {
            for m in val {
                let id = m.landmark_id.expect("recompute_landmark_ids: no landmark id");
                old_max_val = old_max_val.max(id);
                existing_ids.insert(id);
            }
        }

        let mut old_new_map = HashMap::<usize,usize>::with_capacity(old_max_val);
        let mut free_ids = (0..existing_ids.len()).collect::<HashSet<usize>>();

        let mut missing_id_set = (0..old_max_val).collect::<HashSet<usize>>();
        for (_,val) in match_map.iter() {
            for m in val {
                missing_id_set.remove(&m.landmark_id.unwrap());
            }
        }

        for (_,val) in match_map.iter_mut() {
            for m in val {
                let old_id = m.landmark_id.expect("recompute_landmark_ids: no landmark id");
                if old_new_map.contains_key(&old_id) {
                    let new_id = old_new_map.get(&old_id).unwrap();
                    m.landmark_id = Some(*new_id);
                } else {
                    let free_id = free_ids.iter().next().unwrap().clone();
                    free_ids.remove(&free_id);
                    m.landmark_id = Some(free_id);
                    old_new_map.insert(old_id, free_id);
                }
            }
        }
        assert!(free_ids.is_empty());

        let mut validation_set = (0..existing_ids.len()).collect::<HashSet<usize>>();
        for (_,val) in match_map.iter() {
            for m in val {
                validation_set.remove(&m.landmark_id.unwrap());
            }
        }
        assert!(validation_set.is_empty());

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
    fn compute_pose_map(
            root: usize,
            paths: &Vec<Vec<usize>>,
            camera_map: &HashMap<usize, C>,
            mut match_map: HashMap<(usize, usize), Vec<Match<Feat>>>,
            perc_tresh: Float, 
            epipolar_tresh: Float,
            epipolar_alg: tensor::BifocalType) 
        ->  (HashMap<(usize, usize), Isometry3<Float>>,HashMap<(usize, usize), Vec<Match<Feat>>>) {
            let number_of_paths = paths.len();
            let map_capacity = calc_map_capacity(number_of_paths); //TODO expose this in function args
            let mut pose_map = HashMap::<(usize, usize), Isometry3<Float>>::with_capacity(map_capacity);
            for path_idx in 0..number_of_paths {
                let path = paths[path_idx].clone();
                for j in 0..path.len() {
                    let id1 = match j {
                        0 => root,
                        idx => path[idx-1]
                    };
                    let id2 = path[j];
                    let c1 = camera_map.get(&id1).expect("compute_pairwise_cam_motions_for_path: could not get previous cam");
                    let c2 = camera_map.get(&id2).expect("compute_pairwise_cam_motions_for_path: could not get second camera");
                    let key = (id1,id2);
                    let m = match_map.get(&key).expect(format!("match not found with key: {:?}",key).as_str());
                    let (norm_one, norm_one_inv, norm_two, norm_two_inv) = compute_linear_normalization(m);
                    let m_norm = &m.iter().map(|ma| ma.apply_normalisation(&norm_one, &norm_two, -1.0)).collect::<Vec<_>>();
                    let camera_matrix_one = norm_one*c1.get_projection();
                    let camera_matrix_two = norm_two*c2.get_projection();
                    let inverse_camera_matrix_one = c1.get_inverse_projection()*norm_one_inv;
                    let inverse_camera_matrix_two = c2.get_inverse_projection()*norm_two_inv;



                    let f0 = 1.0; // -> check this

                    let (e, f_m,f_m_norm) = match epipolar_alg {
                        tensor::BifocalType::FUNDAMENTAL => {      
                            let f = tensor::fundamental::eight_point_hartley(m_norm, false, f0); //TODO: make this configurable

                            let f_corr = tensor::fundamental::optimal_correction(&f, m_norm, f0);
                            let filtered_indices = tensor::select_best_matches_from_fundamental(&f_corr,m_norm,perc_tresh, epipolar_tresh);

                            //let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh);
                            let filtered = filtered_indices.iter().map(|i| m[*i].clone()).collect::<Vec<Match<Feat>>>();
                            let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();

                            let e = tensor::compute_essential(&f,&camera_matrix_one,&camera_matrix_two);
                            //let e = tensor::compute_essential(&f,&c1.get_projection(),&c2.get_projection());

                            (e, filtered, filtered_norm)
                        },
                        tensor::BifocalType::ESSENTIAL => {
                            //let e = tensor::ransac_five_point_essential(m, c1, c2,1e-2,1e4 as usize);
                            let e = tensor::five_point_essential(m_norm, &camera_matrix_one, &inverse_camera_matrix_one, &camera_matrix_two ,&inverse_camera_matrix_two); //TODO refactor to take matricies instead of camera object
                            let f = tensor::compute_fundamental(&e, &inverse_camera_matrix_one, &inverse_camera_matrix_two);

                            let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh);
                            let filtered = filtered_indices.iter().map(|i| m[*i].clone()).collect::<Vec<Match<Feat>>>();
                            let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();

                            (e, filtered, filtered_norm)
                        },
                        tensor::BifocalType::QUEST => {
                            let e = quest::quest_ransac(m_norm,  &inverse_camera_matrix_one, &inverse_camera_matrix_two, 1e-2,1e4 as usize); //TODO refactor to take matricies instead of camera object
                            let f = tensor::compute_fundamental(&e, &inverse_camera_matrix_one, &inverse_camera_matrix_two);

                            let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh);
                            let filtered = filtered_indices.iter().map(|i| m[*i].clone()).collect::<Vec<Match<Feat>>>();
                            let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();

                            (e, filtered, filtered_norm)
                        }
                    };
                    
                    //TODO subsample?
                    //let f_m_subsampled = subsample_matches(f_m,image_width,image_height);
                    println!("{:?}: Number of matches: {}", key, &f_m.len());

                    
                    // The pose transforms id2 into the coordiante system of id1
                    let (h,rotation,_) = tensor::decompose_essential_förstner(&e,&f_m_norm,&inverse_camera_matrix_two, &inverse_camera_matrix_two);
                    let se3 = se3(&h,&rotation);
                    let isometry = from_matrix(&se3);
                    let some_pose_old_val = pose_map.insert(key, isometry);
                    let some_old_match_val = match_map.insert(key,f_m);
                    assert!(some_pose_old_val.is_none());
                    assert!(!some_old_match_val.is_none());

                }
            }
        (pose_map, match_map)
    }

    fn refine_rotation_by_rcd(root: usize, paths: &Vec<Vec<usize>>, pose_map: &HashMap<(usize, usize), Isometry3<Float>>) -> HashMap<(usize, usize), Isometry3<Float>> {
        let number_of_paths = paths.len(); 
        let mut new_pose_map = HashMap::<(usize, usize), Isometry3<Float>>::with_capacity(pose_map.capacity());
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

                //let initial_rot = initial_pose.rotation.to_rotation_matrix().matrix().to_owned();
                //let angular_distance_initial = angular_distance(&initial_rot);
                //let angular_distance_rcd = angular_distance(&rcd_rot);
                //println!("initial r : {}", initial_rot);
                //println!("rcd r : {}",rcd_rot);
                //println!("initial ang dist : {}", angular_distance_initial);
                //println!("rcd ang dist : {}", angular_distance_rcd);

                let new_se3 = se3(&initial_pose.translation.vector,&rcd_rot);
                new_pose_map.insert(key, from_matrix(&new_se3));
            }
        }

        new_pose_map

    }

}

fn calc_map_capacity(number_of_paths: usize) -> usize {
    100*number_of_paths
}

