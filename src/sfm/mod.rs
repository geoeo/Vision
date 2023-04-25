extern crate nalgebra as na;
extern crate num_traits;


use na::{DVector, Matrix4xX, Vector4, Matrix3, Isometry3};
use std::{collections::{HashMap,HashSet}, hash::Hash};
use crate::image::{features::{Feature, Match, feature_track::FeatureTrack, solver_feature::SolverFeature}};
use crate::sfm::{epipolar::tensor, epipolar::compute_linear_normalization,
    triangulation::{Triangulation, triangulate_matches}, 
    rotation_avg::{optimize_rotations_with_rcd_per_track,optimize_rotations_with_rcd},
    outlier_rejection::outlier_rejection_dual};
use crate::sensors::camera::Camera;
use crate::numerics::{pose::{from_matrix,se3}};
use crate::{float,Float};

pub mod bundle_adjustment;
pub mod landmark;
pub mod epipolar;
pub mod triangulation;
pub mod quest;
pub mod rotation_avg;
pub mod outlier_rejection;

/**
 * We assume that the indices between paths and matches are consistent
 */
pub struct SFMConfig<C, Feat: Feature> {
    root: usize,
    paths: Vec<Vec<usize>>,
    camera_map: HashMap<usize, C>,
    match_map: HashMap<(usize, usize), Vec<Match<Feat>>>, 
    pose_map: HashMap<(usize,usize), Isometry3<Float>>, // The pose transforms tuple id 2 into the coordiante system of tuple id 1
    abs_pose_map: HashMap<usize, Isometry3<Float>>, 
    abs_landmark_map: HashMap<usize, Matrix4xX<Float>>,
    reprojection_error_map: HashMap<(usize, usize),DVector<Float>>,
    unique_landmark_ids: HashSet<usize>,
    epipolar_alg: tensor::BifocalType,
    triangulation: Triangulation
}

pub fn compute_path_pairs_as_vec(root: usize, paths: &Vec<Vec<usize>>) -> Vec<Vec<(usize,usize)>> {
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

pub fn compute_path_id_pairs(root_id: usize, paths: &Vec<Vec<usize>>) -> Vec<Vec<(usize, usize)>> {
    let mut path_id_paris = Vec::<Vec::<(usize,usize)>>::with_capacity(paths.len());
    for sub_path in paths {
        path_id_paris.push(
            sub_path.iter().enumerate().map(|(i,&id)| 
                match i {
                    0 => (root_id,id),
                    idx => (sub_path[idx-1],id)
                }
            ).collect()
        )
    }

    path_id_paris
}

impl<C: Camera<Float>, Feat: Feature + Clone + PartialEq + Eq + Hash + SolverFeature> SFMConfig<C, Feat> {
    pub fn new(root: usize, 
        paths: &Vec<Vec<usize>>, 
        camera_map: HashMap<usize, C>, 
        match_map_no_landmarks: &HashMap<(usize,usize), Vec<Match<Feat>>>, 
        epipolar_alg: tensor::BifocalType, 
        triangulation: Triangulation, 
        perc_tresh: Float, 
        epipolar_thresh: Float, 
        landmark_cutoff_thresh: Float,
        refine_rotation_via_rcd: bool,
        positive_principal_distance: bool) -> SFMConfig<C,Feat> {

        let paths_pairs_as_vec = compute_path_pairs_as_vec(root,paths);
        // Filteres matches according to feature consitency along a path.
        let accepted_matches = Self::filter_by_max_tracks(&paths_pairs_as_vec, &match_map_no_landmarks);
        let found_duplicates = Self::check_for_duplicate_pixel_entries(&accepted_matches);
        assert!(!found_duplicates);
        let match_map = Self::generate_match_map_with_landmark_ids(root, &paths,accepted_matches);

        let (mut pose_map, camera_norm_map, mut match_norm_map) = Self::compute_pose_map_and_normalize(
            camera_map,
            match_map,
            perc_tresh, 
            epipolar_thresh,
            epipolar_alg,
            positive_principal_distance
        );
        let (mut landmark_map, mut reprojection_error_map, min_reprojection_error_initial, max_reprojection_error_initial) 
            = Self::compute_landmarks_and_reprojection_maps(root,&paths,&pose_map,&match_norm_map,&camera_norm_map,triangulation, positive_principal_distance);

        println!("SFM Config Max Reprojection Error 1): {}, Min Reprojection Error: {}", max_reprojection_error_initial, min_reprojection_error_initial);
        Self::reject_landmark_outliers( &mut landmark_map, &mut reprojection_error_map, &mut match_norm_map, landmark_cutoff_thresh);
        if refine_rotation_via_rcd {
            let new_pose_map = Self::refine_rotation_by_rcd(root, &paths, &pose_map);
            let (mut new_landmark_map, mut new_reprojection_error_map , _, _) =  Self::compute_landmarks_and_reprojection_maps(root,&paths,&new_pose_map,&match_norm_map,&camera_norm_map,triangulation, positive_principal_distance);
            let keys = landmark_map.keys().map(|k| *k).collect::<Vec<_>>();
            for key in keys {
                let new_reprojection_errors = new_reprojection_error_map.get(&key).unwrap();
                let current_reprojection_errors = reprojection_error_map.get_mut(&key).unwrap();

                if new_reprojection_errors.mean() < current_reprojection_errors.mean() {
                    landmark_map.insert(key,new_landmark_map.remove(&key).unwrap());
                    reprojection_error_map.insert(key,new_reprojection_error_map.remove(&key).unwrap());
                    pose_map.insert(key,new_pose_map.get(&key).unwrap().clone());
                }
            }
        }

        Self::reject_landmark_outliers(&mut landmark_map, &mut reprojection_error_map, &mut match_norm_map, landmark_cutoff_thresh);
        let (min_reprojection_error_refined, max_reprojection_error_refined) = Self::compute_reprojection_ranges(&reprojection_error_map);
        println!("SFM Config Max Reprojection Error 2): {}, Min Reprojection Error: {}", max_reprojection_error_refined, min_reprojection_error_refined);

        //TODO: Comment recompute_landmark_ids this in more detail
        // Since landmarks may be rejected, this function recomputes the ids to be consecutive so that they may be used for matrix indexing.
        Self::recompute_landmark_ids(&mut match_norm_map);
        let path_id_pairs = compute_path_id_pairs(root, paths);

        let camera_ids_root_first = Self::get_sorted_camera_keys(root, paths);
        let mut unique_landmark_ids = compute_unique_landmark_id(&match_norm_map);
        let mut abs_pose_map = compute_absolute_poses_for_root(root, &path_id_pairs, &pose_map);
        let (mut feature_map, landmark_id_cam_pair_index_map) = compute_features_per_image_map(&match_norm_map, &unique_landmark_ids); 
        let mut abs_landmark_map = compute_absolute_landmarks_for_root(&path_id_pairs,&landmark_map,&abs_pose_map);
        let root_cam = camera_norm_map.get(&root).expect("Root Cam not found!");
        let tol = 5.0/root_cam.get_focal_x(); // rougly 5 pixels
        //outlier_rejection_dual(&camera_ids_root_first, &mut unique_landmark_ids, &mut abs_landmark_map, &mut abs_pose_map, &mut feature_map, &mut match_norm_map, &landmark_id_cam_pair_index_map, tol);


        SFMConfig{root, paths: paths.clone(), camera_map: camera_norm_map, match_map: match_norm_map, abs_pose_map, pose_map, epipolar_alg, abs_landmark_map, reprojection_error_map, unique_landmark_ids, triangulation}
    }


    pub fn root(&self) -> usize { self.root }
    pub fn paths(&self) -> &Vec<Vec<usize>> { &self.paths }
    pub fn camera_map_highp(&self) -> &HashMap<usize, C> { &self.camera_map}
    pub fn epipolar_alg(&self) -> tensor::BifocalType { self.epipolar_alg}
    pub fn triangulation(&self) -> Triangulation { self.triangulation}
    pub fn match_map(&self) -> &HashMap<(usize, usize), Vec<Match<Feat>>> {&self.match_map}
    pub fn abs_pose_map(&self) -> &HashMap<usize, Isometry3<Float>> {&self.abs_pose_map}
    pub fn pose_map(&self) -> &HashMap<(usize,usize), Isometry3<Float>> {&self.pose_map}
    pub fn abs_landmark_map(&self) -> &HashMap<usize, Matrix4xX<Float>> {&self.abs_landmark_map}
    pub fn reprojection_error_map(&self) -> &HashMap<(usize, usize), DVector<Float>> {&self.reprojection_error_map}
    pub fn unique_landmark_ids(&self) -> &HashSet<usize> {&self.unique_landmark_ids}

    pub fn compute_unqiue_ids_cameras_root_first(&self) -> (Vec<usize>, Vec<&C>) {
        let keys_sorted = Self::get_sorted_camera_keys(self.root(), self.paths());
        let cameras_sorted = keys_sorted.iter().map(|id| self.camera_map.get(id).expect("compute_unqiue_ids_cameras_root_first: trying to get invalid camera")).collect::<Vec<&C>>();
        (keys_sorted, cameras_sorted)
    }

    fn check_for_duplicate_pixel_entries(matches: &Vec<Vec<Vec<Match<Feat>>>>) -> bool {
        let mut duplicates_found = false;
        for path in matches{
            for tracks in path {
                let mut pixel_map = HashMap::<(usize,usize), (Float, Float,Float, Float)>::with_capacity(1000*1000);
                for m in tracks {
                    let f = m.get_feature_one().get_as_2d_point();
                    let f2 = m.get_feature_two().get_as_2d_point();
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

    fn generate_match_map_with_landmark_ids(root: usize, paths: &Vec<Vec<usize>>, matches: Vec<Vec<Vec<Match<Feat>>>>) -> HashMap<(usize,usize), Vec<Match<Feat>>> {
        let number_of_paths = paths.len();
        let match_vec_capacity = 500; //TODO: Set this in a better way
        let mut match_map = HashMap::<(usize, usize), Vec<Match<Feat>>>::with_capacity(match_vec_capacity);
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
            let mut rejected_landmark_ids = HashSet::<usize>::with_capacity(1000);

            for key in &keys {
                let reprojection_erros = reprojection_error_map.get(key).unwrap();
                let matches = match_map.get(key).unwrap();
                
                let rejected_indices = reprojection_erros.iter().enumerate().filter(|&(_,v)| *v >= landmark_cutoff).map(|(idx,_)| idx).collect::<HashSet<usize>>();
                rejected_landmark_ids.extend(matches.iter().enumerate().filter(|(idx,_)|rejected_indices.contains(idx)).map(|(_,v)| v.get_landmark_id().unwrap()).collect::<HashSet<_>>());
            }

            for key in &keys {
                let reprojection_erros = reprojection_error_map.get(key).unwrap();
                let matches = match_map.get(key).unwrap();
                let landmarks = landmark_map.get(key).unwrap();

                let accepted_indices = matches.iter().enumerate().filter(|&(_,v)| !rejected_landmark_ids.contains(&v.get_landmark_id().unwrap())).map(|(idx,_)| idx).collect::<HashSet<usize>>();
                let filtered_matches = matches.iter().enumerate().filter(|(idx,_)|accepted_indices.contains(idx)).map(|(_,v)| v.clone()).collect::<Vec<_>>();
                assert!(!&filtered_matches.is_empty(), "reject outliers empty features for : {:?}", key);

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
        triangulation: Triangulation,
        positive_principal_distance: bool) -> (HashMap<(usize,usize),Matrix4xX<Float>>, HashMap<(usize,usize), DVector<Float>>, Float, Float) {

        let mut triangulated_match_map = HashMap::<(usize,usize),Matrix4xX<Float>>::with_capacity(match_map.len());
        let mut reprojection_map = HashMap::<(usize,usize),DVector<Float>>::with_capacity(match_map.len());
        let path_pairs = compute_path_pairs_as_vec(root,paths);
        let mut max_reprojection_error = float::MIN;
        let mut min_reprojection_error = float::MAX;

        for path in &path_pairs{
            for path_pair in path {
                let (trigulated_matches,reprojection_errors) = triangulate_matches(*path_pair,&pose_map,&match_map,&camera_map,triangulation, positive_principal_distance);
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

    fn get_sorted_camera_keys(root_id: usize, paths: &Vec<Vec<usize>>) -> Vec<usize> {
        let ids_flat = paths.clone().into_iter().flatten().collect::<Vec<usize>>();
        let number_of_keys = ids_flat.len() + 1;
        let mut keys_sorted = Vec::<usize>::with_capacity(number_of_keys);
        // root has to first by design
        keys_sorted.push(root_id);
        keys_sorted.extend(ids_flat);
        keys_sorted.dedup();
        keys_sorted
    }

    //TODO: merges in tracks originating at the root
    fn filter_by_max_tracks(all_paths: &Vec<Vec<(usize,usize)>>, match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>) -> Vec<Vec<Vec<Match<Feat>>>> {

        let mut filtered_matches = Vec::<Vec<Vec<Match<Feat>>>>::with_capacity(all_paths.len()); 
        let mut feature_tracks = Vec::<Vec<FeatureTrack<Feat>>>::with_capacity(all_paths.len());

        for path_idx in 0..all_paths.len() {
            let path = &all_paths[path_idx];
            let path_len = path.len();
            filtered_matches.push(Vec::<Vec<Match<Feat>>>::with_capacity(path_len));
            feature_tracks.push(Vec::<FeatureTrack<Feat>>::with_capacity(path_len));
        
            for pair_idx in 0..path_len {
                let k = path[pair_idx];
                filtered_matches[path_idx].push(Vec::<Match<Feat>>::with_capacity(match_map.get(&k).unwrap().len()));
            }
        }

        let mut landmark_id = 0;
        let max_path_len: usize = all_paths.iter().map(|x| x.len()).sum();
        for path_idx in 0..all_paths.len() {
            let path_pairs_for_path = all_paths[path_idx].clone();
            let path_len = path_pairs_for_path.len();
            for img_idx in 0..path_len {
                let current_pair_key = &path_pairs_for_path[img_idx].clone();
                let current_matches = match_map.get(current_pair_key).unwrap();
                let mut pixel_set = HashSet::<(usize,usize)>::with_capacity(current_matches.len());
                for m in current_matches {
                    let f1 = m.get_feature_one();
                    let f1_x = f1.get_x_image();
                    let f1_y = f1.get_y_image();
                    let k = (f1_x,f1_y);
                    match img_idx {
                        0 => {
                            if !pixel_set.contains(&k) {
                                let mut id : Option<usize> = None;
                                for j in 0..feature_tracks.len() {
                                    for i in 0..feature_tracks[j].len() {
                                        let track = &feature_tracks[j][i];
                                        if track.get_first_feature_start() == f1.clone() {
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
                            let current_feature_one = f1;
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
                let id = m.get_landmark_id().expect("recompute_landmark_ids: no landmark id");
                old_max_val = old_max_val.max(id);
                existing_ids.insert(id);
            }
        }

        let mut old_new_map = HashMap::<usize,usize>::with_capacity(old_max_val);
        let mut free_ids = (0..existing_ids.len()).collect::<HashSet<usize>>();

        let mut missing_id_set = (0..old_max_val).collect::<HashSet<usize>>();
        for (_,val) in match_map.iter() {
            for m in val {
                missing_id_set.remove(&m.get_landmark_id().unwrap());
            }
        }

        for (_,val) in match_map.iter_mut() {
            for m in val {
                let old_id = m.get_landmark_id().expect("recompute_landmark_ids: no landmark id");
                if old_new_map.contains_key(&old_id) {
                    let new_id = old_new_map.get(&old_id).unwrap();
                    m.set_landmark_id(Some(*new_id));
                } else {
                    let free_id = free_ids.iter().next().unwrap().clone();
                    free_ids.remove(&free_id);
                    m.set_landmark_id(Some(free_id));
                    old_new_map.insert(old_id, free_id);
                }
            }
        }
        assert!(free_ids.is_empty());

        let mut validation_set = (0..existing_ids.len()).collect::<HashSet<usize>>();
        for (_,val) in match_map.iter() {
            for m in val {
                validation_set.remove(&m.get_landmark_id().unwrap());
            }
        }
        assert!(validation_set.is_empty());

    }

    //TODO: move normalization into a separate function
    #[allow(non_snake_case)]
    fn compute_pose_map_and_normalize(
            camera_map: HashMap<usize, C>,
            match_map: HashMap<(usize, usize), Vec<Match<Feat>>>,
            perc_tresh: Float, 
            epipolar_tresh: Float,
            epipolar_alg: tensor::BifocalType,
            positive_principal_distance: bool) 
        ->  (HashMap<(usize, usize), Isometry3<Float>>,HashMap<usize, C>,HashMap<(usize, usize), Vec<Match<Feat>>>) {
            let mut pose_map = HashMap::<(usize, usize), Isometry3<Float>>::with_capacity(match_map.len());
            let match_keys : Vec<_> = match_map.iter().map(|(k,_)| *k).collect();

            let mut feature_map = HashMap::<usize, Vec<Feat>>::with_capacity(camera_map.len());
            let mut normalization_map = HashMap::<usize, Matrix3<Float>>::with_capacity(camera_map.len());
            let mut camera_norm_map =  HashMap::<usize, C>::with_capacity(camera_map.len());
            let mut match_norm_map =  HashMap::<(usize, usize), Vec<Match<Feat>>>::with_capacity(match_map.len());


            for key in camera_map.keys() {
                //TODO: Better size estimation
                feature_map.insert(*key, Vec::<Feat>::with_capacity(500));
            }

            for ((id1,id2),matches) in match_map.iter() {
                feature_map.get_mut(id1).expect("compute_pose_map: camera doesnt exist").extend(matches.iter().map(|m| m.get_feature_one().clone()));
                feature_map.get_mut(id2).expect("compute_pose_map: camera doesnt exist").extend(matches.iter().map(|m| m.get_feature_two().clone()));
                match_norm_map.insert((*id1,*id2), Vec::<Match<Feat>>::with_capacity(matches.len()));
            }

            for (cam_id, features) in feature_map.iter() {
                let (norm, norm_inv) = compute_linear_normalization(features);
                let c = camera_map.get(cam_id).expect("compute_pose_map: cam not found");
                let camera_matrix = norm*c.get_projection();
                let inverse_camera_matrix = c.get_inverse_projection()*norm_inv;
                let c_norm: C = Camera::from_matrices(&camera_matrix, &inverse_camera_matrix);
                normalization_map.insert(*cam_id,norm);
                camera_norm_map.insert(*cam_id, c_norm);
            }

            for (id1,id2) in match_keys {
                let c1 = camera_norm_map.get(&id1).expect("compute_pose_map: could not get previous cam");
                let c2 = camera_norm_map.get(&id2).expect("compute_pose_map: could not get second camera");
                let key = (id1,id2);
                let m = match_map.get(&key).expect(format!("match not found with key: {:?}",key).as_str());

                let norm_one = normalization_map.get(&id1).expect("compute_pose_map: normalization matrix not found");
                let norm_two = normalization_map.get(&id2).expect("compute_pose_map: normalization matrix not found");
                let m_norm = &m.iter().map(|ma| ma.apply_normalisation(&norm_one, &norm_two, 1.0)).collect::<Vec<_>>();
                let camera_matrix_one = c1.get_projection();
                let camera_matrix_two = c2.get_projection();
                let inverse_camera_matrix_one = c1.get_inverse_projection();
                let inverse_camera_matrix_two = c2.get_inverse_projection();

                let (e,f_m_norm) = match epipolar_alg {
                    tensor::BifocalType::FUNDAMENTAL => {      
                        let f = tensor::fundamental::eight_point_hartley(m_norm, 1.0); 
                        
                        let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh, 1.0);
                        let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();

                        let e = tensor::compute_essential(&f,&camera_matrix_one,&camera_matrix_two);

                        (e, filtered_norm)
                    },
                    tensor::BifocalType::ESSENTIAL => {
                        let e = tensor::five_point_essential(m_norm, &camera_matrix_one, &inverse_camera_matrix_one, &camera_matrix_two ,&inverse_camera_matrix_two, positive_principal_distance); 
                        let f = tensor::compute_fundamental(&e, &inverse_camera_matrix_one, &inverse_camera_matrix_two);

                        let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh, 1.0);
                        let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();

                        (e, filtered_norm)
                    },
                    tensor::BifocalType::QUEST => {
                        let e = quest::quest_ransac(m_norm,  &inverse_camera_matrix_one, &inverse_camera_matrix_two, 1e-2,1e4 as usize, positive_principal_distance); 
                        let f = tensor::compute_fundamental(&e, &inverse_camera_matrix_one, &inverse_camera_matrix_two);

                        let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh, 1.0);
                        let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();

                        (e, filtered_norm)
                    }
                };
                
                //TODO subsample?
                //let f_m_subsampled = subsample_matches(f_m,image_width,image_height);
                println!("{:?}: Number of matches: {}", key, &f_m_norm.len());
                // The pose transforms id2 into the coordiante system of id1
                let (iso3_opt,_) = tensor::decompose_essential_förstner(&e,&f_m_norm,&inverse_camera_matrix_two, &inverse_camera_matrix_two,positive_principal_distance);
                let _ = match iso3_opt {
                    Some(isometry) => pose_map.insert(key, isometry),
                    None => {
                        println!("Warning: Decomposition of essential matrix failed for pair ({},{})",id1,id2);
                        pose_map.insert(key, Isometry3::<Float>::identity())
                    }
                };
                let _ = match_norm_map.insert(key,f_m_norm);
            }

        (pose_map, camera_norm_map, match_norm_map)
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
        let initial_cam_rotations_per_path_rcd = optimize_rotations_with_rcd(&initial_cam_motions_per_path);
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

fn compute_absolute_poses_for_root(root: usize, paths: &Vec<Vec<(usize,usize)>>, pose_map: &HashMap<(usize, usize), Isometry3<Float>>) -> HashMap<usize,Isometry3<Float>> {
    let flattened_path_len = paths.iter().flatten().collect::<Vec<_>>().len();
    let mut abs_pose_map = HashMap::<usize,Isometry3<Float>>::with_capacity(flattened_path_len+1);
    abs_pose_map.insert(root, Isometry3::<Float>::identity());

    for path in paths {
        let mut pose_acc = Isometry3::<Float>::identity();
        for key in path {
            let pose = pose_map.get(key).expect("Error in compute_absolute_poses_for_root: Pose for key not found ");
            pose_acc *= pose;
            abs_pose_map.insert(key.1,pose_acc);
        }
    }
    abs_pose_map
}

fn compute_absolute_landmarks_for_root(paths: &Vec<Vec<(usize,usize)>>, landmark_map: &HashMap<(usize, usize), Matrix4xX<Float>>, abs_pose_map: &HashMap<usize,Isometry3<Float>>) -> HashMap<usize,Matrix4xX<Float>> {
    let flattened_path_len = paths.iter().flatten().collect::<Vec<_>>().len();
    let mut abs_landmark_map = HashMap::<usize,Matrix4xX<Float>>::with_capacity(flattened_path_len+1);
    for path in paths {
        for (id_s, id_f) in path {
            let landmark_key = (*id_s, *id_f);
            let triangulated_matches = landmark_map.get(&landmark_key).expect(format!("no landmarks found for key: {:?}",landmark_key).as_str());
            let pose = abs_pose_map.get(id_f).expect("compute_absolute_landmarks_for_root: pose not found").to_matrix();
            let root_aligned_triangulated_matches = pose*triangulated_matches;
            abs_landmark_map.insert(*id_f,root_aligned_triangulated_matches);
        }
    }
    abs_landmark_map
}

fn compute_unique_landmark_id<Feat: Feature>(match_map: &HashMap<(usize,usize), Vec<Match<Feat>>>) -> HashSet<usize> {
    let mut unique_landmarks_ids = HashSet::<usize>::new();
    for ms in match_map.values() {
        for m in ms {
            unique_landmarks_ids.insert(m.get_landmark_id().unwrap());
        }
    }
    unique_landmarks_ids
}

fn compute_features_per_image_map<Feat: Feature + Clone>(match_map: &HashMap<(usize,usize), Vec<Match<Feat>>>, unique_landmark_ids: &HashSet<usize>) -> (HashMap<usize, Vec<Feat>>, HashMap<usize,Vec<((usize,usize),usize)>>) {
    let mut unique_cameras = HashSet::<usize>::new();
    let mut landmark_id_cam_pair_index_map = HashMap::<usize,Vec<((usize,usize),usize)>>::with_capacity(unique_landmark_ids.len());

    for key in match_map.keys() {
        unique_cameras.insert(key.0);
        unique_cameras.insert(key.1);
    }

    let features_per_landmark = unique_cameras.len()*unique_landmark_ids.len()*0.4 as usize; //TODO: Better heuristic
    for landmark_id in unique_landmark_ids {
        landmark_id_cam_pair_index_map.insert(*landmark_id, Vec::<((usize,usize),usize)>::with_capacity(features_per_landmark));
    }
    
    let mut feature_map = HashMap::<usize, Vec<Feat>>::with_capacity(unique_cameras.len());
    for unique_cam_id in &unique_cameras {
        feature_map.insert(unique_cam_id.clone(),  Vec::<Feat>::with_capacity(unique_landmark_ids.len())); 
    }

    for camera_id in &unique_cameras {
        let id_prev = match camera_id {
            0 => None,
            i => Some(i-1)
        };
        let id_next = camera_id + 1;
        let cam_pair_fwd_key = (camera_id.clone(),id_next);
        let features_fwd = match_map.get(&cam_pair_fwd_key);
        if features_fwd.is_some(){
            let matches = features_fwd.unwrap();
            for vec_idx in 0..matches.len() {
                let m = &matches[vec_idx];
                let f_1 = m.get_feature_one();
                let f_2 = m.get_feature_two();
                let landmark_id = f_1.get_landmark_id().expect("compute_features_per_image_map: fwd landmark id not found");
                feature_map.get_mut(&camera_id).expect("compute_features_per_image_map: Camera not found in bck feature").push(f_1.clone());
                feature_map.get_mut(&id_next).expect("compute_features_per_image_map: Camera not found in bck feature").push(f_2.clone());
                landmark_id_cam_pair_index_map.get_mut(&landmark_id).unwrap().push((cam_pair_fwd_key,vec_idx));
            }
        }
        
        if id_prev.is_some() {
            let id_prev_val = id_prev.unwrap();
            let cam_pair_bck_key = (camera_id.clone(),id_prev_val);
            let features_bck = match_map.get(&cam_pair_bck_key);
            if features_bck.is_some(){
                let matches = features_bck.unwrap();
                for vec_idx in 0..matches.len() {
                    let m = &matches[vec_idx];
                    let f_1 = m.get_feature_one();
                    let f_2 = m.get_feature_two();
                    let landmark_id = f_2.get_landmark_id().expect("compute_features_per_image_map: bck landmark id not found");
                    landmark_id_cam_pair_index_map.get_mut(&landmark_id).unwrap().push((cam_pair_bck_key,vec_idx));
                    feature_map.get_mut(&camera_id).expect("compute_features_per_image_map: Camera not found in bck feature").push(f_1.clone());
                    feature_map.get_mut(&id_prev_val).expect("compute_features_per_image_map: Camera not found in bck feature").push(f_2.clone());
                }
            }
        }
    }

    (feature_map, landmark_id_cam_pair_index_map)
}

