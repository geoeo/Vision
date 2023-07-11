extern crate nalgebra as na;
extern crate num_traits;


use na::{DVector, Matrix4xX, Matrix3, Matrix4, Isometry3};
use std::{collections::{HashMap,HashSet}, hash::Hash};
use crate::image::{features::{Feature, compute_linear_normalization, matches::Match, feature_track::FeatureTrack, solver_feature::SolverFeature}};
use crate::sfm::{epipolar::tensor, 
    triangulation::{Triangulation, triangulate_matches}, 
    rotation_avg::optimize_rotations_with_rcd,
    outlier_rejection::dual::outlier_rejection_dual};
use crate::sfm::outlier_rejection::{calculate_reprojection_errors,calcualte_disparities, reject_landmark_outliers, filter_by_rejected_landmark_ids, reject_matches_via_disparity, compute_continuous_landmark_ids_from_matches,compute_continuous_landmark_ids_from_unique_landmarks};
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
    match_norm_map: HashMap<(usize, usize), Vec<Match<Feat>>>, 
    pose_map: HashMap<(usize,usize), Isometry3<Float>>, // The pose transforms tuple id 2 into the coordiante system of tuple id 1
    abs_pose_map: HashMap<usize, Isometry3<Float>>, 
    abs_landmark_map: HashMap<usize, Matrix4xX<Float>>,
    reprojection_error_map: HashMap<(usize, usize),DVector<Float>>,
    unique_landmark_ids: HashSet<usize>,
    epipolar_alg: tensor::BifocalType,
    triangulation: Triangulation
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
        disparity_cutoff_thresh: Float,
        refine_rotation_via_rcd: bool,
        run_outlier_detection_pipeline: bool,
        positive_principal_distance: bool) -> SFMConfig<C,Feat> {

        let paths_pairs_as_vec = compute_path_pairs_as_vec(root,paths);
        let camera_ids_root_first = Self::get_sorted_camera_keys(root, paths);

        //TODO: Compute Image Score for later filtering

        // Filteres matches according to feature consitency along a path.
        let accepted_matches = Self::filter_by_max_tracks(&paths_pairs_as_vec, &match_map_no_landmarks);
        let found_duplicates = Self::check_for_duplicate_pixel_entries(&accepted_matches);
        assert!(!found_duplicates);
        let mut match_map = Self::generate_match_map_with_landmark_ids(root, &paths,accepted_matches);

        let disparity_map = Self::compute_disparity_map(root,&paths,&match_map);
        reject_matches_via_disparity(disparity_map, &mut match_map, disparity_cutoff_thresh);

        let (camera_norm_map, mut match_norm_map) = Self::normalize_features_and_cameras(&camera_map, &match_map);
        let mut pose_map = Self::compute_initial_pose_map_and_filter_matches(
            &mut match_map,
            &mut match_norm_map,
            &camera_norm_map,
            perc_tresh, 
            epipolar_thresh,
            epipolar_alg,
            positive_principal_distance);
            
        let (_,mut unique_landmark_ids) = compute_continuous_landmark_ids_from_matches(&mut match_norm_map, &mut match_map,None, None);
        let (mut landmark_map, mut reprojection_error_map) 
            = Self::compute_landmarks_and_reprojection_maps(root,&paths,&pose_map,&match_norm_map,&camera_norm_map,triangulation, positive_principal_distance);
        let path_id_pairs = compute_path_id_pairs(root, paths);
        let path_id_pairs_flat = path_id_pairs.iter().flatten().collect::<Vec<_>>();
        let mut abs_pose_map = compute_absolute_poses_for_root(root, &path_id_pairs, &pose_map);
        let mut abs_landmark_map = compute_absolute_landmarks_for_root(&path_id_pairs,&landmark_map,&abs_pose_map);


        
        if run_outlier_detection_pipeline {
            let (min_reprojection_error_initial, max_reprojection_error_initial) = Self::compute_reprojection_ranges(&reprojection_error_map);
            println!("SFM Config Max Reprojection Error 1): {}, Min Reprojection Error: {}", max_reprojection_error_initial, min_reprojection_error_initial);
    

            let tol = 10.0/camera_map.get(&root).expect("Root Cam missing").get_focal_x(); // rougly 5 pixels //TODO expose this
            //Self::filter_outliers_by_dual(tol,&camera_ids_root_first,&mut unique_landmark_ids,&mut abs_pose_map,&mut abs_landmark_map,&mut match_norm_map,&mut match_map, &mut landmark_map, &mut reprojection_error_map);
            Self::filter_outliers_by_dual_pairwise(tol,&path_id_pairs_flat,&camera_ids_root_first , &mut unique_landmark_ids,&mut abs_pose_map,&mut abs_landmark_map,&mut match_norm_map,&mut match_map, &mut landmark_map, &mut reprojection_error_map);
    
            reject_landmark_outliers(&mut landmark_map, &mut reprojection_error_map ,&mut match_map ,&mut match_norm_map, landmark_cutoff_thresh);
            if refine_rotation_via_rcd {
                let new_pose_map = Self::refine_rotation_by_rcd(root, &paths, &pose_map);
                let (mut new_landmark_map, mut new_reprojection_error_map) 
                    = Self::compute_landmarks_and_reprojection_maps(root,&paths,&new_pose_map, &match_norm_map, &camera_norm_map,triangulation, positive_principal_distance);
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
                reject_landmark_outliers(&mut landmark_map, &mut reprojection_error_map, &mut match_map, &mut match_norm_map, landmark_cutoff_thresh);
            }
    
            let (min_reprojection_error_refined, max_reprojection_error_refined) = Self::compute_reprojection_ranges(&reprojection_error_map);
            println!("SFM Config Max Reprojection Error 2): {}, Min Reprojection Error: {}", max_reprojection_error_refined, min_reprojection_error_refined);
    
            // Since landmarks may be rejected, this function recomputes the ids to be consecutive so that they may be used for matrix indexing.
            let (_, unique_landmark_ids) = compute_continuous_landmark_ids_from_matches(&mut match_norm_map, &mut match_map,Some(&unique_landmark_ids), None);
        }

        
        SFMConfig{root, paths: paths.clone(), camera_map: camera_norm_map, match_map, match_norm_map, abs_pose_map, pose_map, epipolar_alg, abs_landmark_map, reprojection_error_map, unique_landmark_ids, triangulation}
    }


    pub fn root(&self) -> usize { self.root }
    pub fn paths(&self) -> &Vec<Vec<usize>> { &self.paths }
    pub fn camera_map_highp(&self) -> &HashMap<usize, C> { &self.camera_map}
    pub fn epipolar_alg(&self) -> tensor::BifocalType { self.epipolar_alg}
    pub fn triangulation(&self) -> Triangulation { self.triangulation}
    pub fn match_norm_map(&self) -> &HashMap<(usize, usize), Vec<Match<Feat>>> {&self.match_norm_map}
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

    fn compute_landmarks_and_reprojection_maps(
        root: usize, 
        paths: &Vec<Vec<usize>>, 
        pose_map: &HashMap<(usize, usize), Isometry3<Float>>, 
        match_map: &HashMap<(usize, usize),Vec<Match<Feat>>>, 
        camera_map: &HashMap<usize, C>,
        triangulation: Triangulation,
        positive_principal_distance: bool) -> (HashMap<(usize,usize),Matrix4xX<Float>>, HashMap<(usize,usize), DVector<Float>>) {

        let mut triangulated_match_map = HashMap::<(usize,usize),Matrix4xX<Float>>::with_capacity(match_map.len());
        let mut reprojection_map = HashMap::<(usize,usize),DVector<Float>>::with_capacity(match_map.len());

        let path_pairs = compute_path_pairs_as_vec(root,paths);
        for path in &path_pairs{
            for path_pair in path {
                let trigulated_matches = triangulate_matches(*path_pair,&pose_map,&match_map,&camera_map,triangulation, positive_principal_distance);

                let se3 = pose_map.get(path_pair).expect(format!("compute_landmarks_and_reprojection_maps: pose not found with key: {:?}",path_pair).as_str()).to_matrix();
                let ms = match_map.get(path_pair).expect(format!("compute_landmarks_and_reprojection_maps: matches not found with key: {:?}",path_pair).as_str());
                let cam_1 = camera_map.get(&path_pair.0).expect("compute_landmarks_and_reprojection_maps: camera 1 not found");
                let cam_2 = camera_map.get(&path_pair.1).expect("compute_landmarks_and_reprojection_maps: camera 2 not found");
                let transform_c1 = Matrix4::<Float>::identity().fixed_view::<3,4>(0,0).into_owned();
                let transform_c2 = se3.fixed_view::<3,4>(0,0).into_owned();
                let reprojection_errors = calculate_reprojection_errors(&trigulated_matches, ms, &transform_c1, cam_1, &transform_c2, cam_2);

                triangulated_match_map.insert(*path_pair,trigulated_matches);
                reprojection_map.insert(*path_pair,reprojection_errors);
                
            }
        }

        (triangulated_match_map, reprojection_map)
    }

    fn compute_disparity_map(root: usize, paths: &Vec<Vec<usize>>, match_map: &HashMap<(usize, usize),Vec<Match<Feat>>>) -> HashMap::<(usize,usize),DVector<Float>> {
        let mut disparity_map = HashMap::<(usize,usize),DVector<Float>>::with_capacity(match_map.len());
        let path_pairs = compute_path_pairs_as_vec(root,paths);
        for path in &path_pairs{
            for path_pair in path {
                let ms = match_map.get(path_pair).expect(format!("compute_disparity_maps: matches not found with key: {:?}",path_pair).as_str());
                let disparities = calcualte_disparities(ms);
                disparity_map.insert(*path_pair,disparities);
            }
        }
        disparity_map
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

        //let max_tracks: Vec<Vec<FeatureTrack<Feat>>> = feature_tracks.into_iter().zip(max_track_lengths).map(| (xs, max) | xs.into_iter().filter(|x| x.get_track_length() == max).collect()).collect();
        //let max_tracks: Vec<Vec<FeatureTrack<Feat>>> = feature_tracks.into_iter().zip(max_track_lengths).map(| (xs, max) | xs.into_iter().filter(|x| (x.get_track_length() == max) || (x.get_track_length() == max -1)).collect()).collect();
        let max_tracks = feature_tracks;

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

    fn normalize_features_and_cameras(camera_map: &HashMap<usize, C>,match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>) 
        -> (HashMap<usize, C>,HashMap<(usize, usize), Vec<Match<Feat>>>) {
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

        let match_keys : Vec<_> = match_map.iter().map(|(k,_)| *k).collect();
        for (id1,id2) in match_keys {
                let key = (id1,id2);
                let m = match_map.get(&key).expect(format!("match not found with key: {:?}",key).as_str());

                let norm_one = normalization_map.get(&id1).expect("compute_pose_map: normalization matrix not found");
                let norm_two = normalization_map.get(&id2).expect("compute_pose_map: normalization matrix not found");
                let m_norm = m.iter().map(|ma| ma.apply_normalisation(&norm_one, &norm_two, 1.0)).collect::<Vec<_>>();
                match_norm_map.insert((id1,id2), m_norm);
        }
        (camera_norm_map, match_norm_map)
    }


    #[allow(non_snake_case)]
    fn compute_initial_pose_map_and_filter_matches(
            match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
            match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
            camera_norm_map: &HashMap<usize, C>,
            perc_tresh: Float, 
            epipolar_tresh: Float,
            epipolar_alg: tensor::BifocalType,
            positive_principal_distance: bool) 
        ->  HashMap<(usize, usize), Isometry3<Float>> {
            let mut pose_map = HashMap::<(usize, usize), Isometry3<Float>>::with_capacity(match_norm_map.len());
            let match_keys : Vec<_> = match_norm_map.iter().map(|(k,_)| *k).collect();

            for (id1,id2) in match_keys {
                let c1 = camera_norm_map.get(&id1).expect("compute_pose_map: could not get previous cam");
                let c2 = camera_norm_map.get(&id2).expect("compute_pose_map: could not get second camera");
                let key = (id1,id2);
                let m_norm = match_norm_map.get(&key).expect(format!("norm match not found with key: {:?}",key).as_str());
                let m = match_map.get(&key).expect(format!("match not found with key: {:?}",key).as_str());

                let camera_matrix_one = c1.get_projection();
                let camera_matrix_two = c2.get_projection();
                let inverse_camera_matrix_one = c1.get_inverse_projection();
                let inverse_camera_matrix_two = c2.get_inverse_projection();

                let (e,f_m_norm, f_m) = match epipolar_alg {
                    tensor::BifocalType::FUNDAMENTAL => {      
                        let f = tensor::fundamental::eight_point_hartley(m_norm, 1.0); 
                        
                        let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh, 1.0);
                        let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();
                        let filtered = filtered_indices.iter().map(|i| m[*i].clone()).collect::<Vec<Match<Feat>>>();

                        let e = tensor::compute_essential(&f,&camera_matrix_one,&camera_matrix_two);

                        (e, filtered_norm, filtered)
                    },
                    tensor::BifocalType::ESSENTIAL => {
                        let e = tensor::five_point_essential(m_norm, &camera_matrix_one, &inverse_camera_matrix_one, &camera_matrix_two ,&inverse_camera_matrix_two, positive_principal_distance); 
                        //let e = tensor::ransac_five_point_essential(m_norm, &camera_matrix_one, &inverse_camera_matrix_one, &camera_matrix_two ,&inverse_camera_matrix_two,1e-2,5e4 as usize, positive_principal_distance); 
                        let f = tensor::compute_fundamental(&e, &inverse_camera_matrix_one, &inverse_camera_matrix_two);

                        let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh, 1.0);
                        let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();
                        let filtered = filtered_indices.iter().map(|i| m[*i].clone()).collect::<Vec<Match<Feat>>>();

                        (e, filtered_norm, filtered)
                    },
                    tensor::BifocalType::QUEST => {
                        let e = quest::quest_ransac(m_norm,  &inverse_camera_matrix_one, &inverse_camera_matrix_two, 1e-2,1e4 as usize); 
                        let f = tensor::compute_fundamental(&e, &inverse_camera_matrix_one, &inverse_camera_matrix_two);

                        let filtered_indices = tensor::select_best_matches_from_fundamental(&f,m_norm,perc_tresh, epipolar_tresh, 1.0);
                        let filtered_norm = filtered_indices.iter().map(|i| m_norm[*i].clone()).collect::<Vec<Match<Feat>>>();
                        let filtered = filtered_indices.iter().map(|i| m[*i].clone()).collect::<Vec<Match<Feat>>>();

                        (e, filtered_norm, filtered)
                    }
                };
                
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
                let _ = match_map.insert(key,f_m);
            }

        pose_map
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

    fn filter_outliers_by_dual(
        tol: Float,
        unique_camera_ids_root_first: &Vec<usize>,
        unique_landmark_ids: &mut HashSet<usize>,
        abs_pose_map: &mut HashMap<usize,Isometry3<Float>>,
        abs_landmark_map: &mut HashMap::<usize,Matrix4xX<Float>>,
        match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>, 
        match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        landmark_map: &mut  HashMap<(usize, usize), Matrix4xX<Float>>, 
        reprojection_error_map: &mut HashMap<(usize, usize),DVector<Float>>
    ) -> () {

        let mut feature_map = compute_features_per_image_map(&match_norm_map, &unique_landmark_ids,unique_camera_ids_root_first); 
        let rejected_landmark_ids = outlier_rejection_dual(unique_camera_ids_root_first, unique_landmark_ids,abs_pose_map, &feature_map, tol);
        if !rejected_landmark_ids.is_empty() {
            //TODO: Check if reprojections actually decreased since we have to theoretical guarantee
            filter_by_rejected_landmark_ids(&rejected_landmark_ids, unique_landmark_ids, abs_landmark_map,  match_norm_map,  match_map, landmark_map, &mut feature_map, reprojection_error_map);
        }
    }

    pub fn filter_outliers_by_dual_pairwise(
        tol: Float,
        path_pairs: &Vec<&(usize, usize)>,
        unique_camera_ids: &Vec<usize>, 
        unique_landmark_ids: &mut HashSet<usize>, 
        abs_pose_map: &HashMap<usize,Isometry3<Float>>, 
        abs_landmark_map: &mut HashMap::<usize,Matrix4xX<Float>>,
        match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>, 
        match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        landmark_map: &mut  HashMap<(usize, usize), Matrix4xX<Float>>, 
        reprojection_error_map: &mut HashMap<(usize, usize),DVector<Float>>
    ) -> () {
    let mut feature_map = compute_features_per_image_map(&match_norm_map, &unique_landmark_ids, unique_camera_ids); 

    let rejected_landmark_ids = path_pairs.clone().into_iter().map(|&(new_root,new_target)| {
        let new_camera_ids_root_first = vec![new_root,new_target];
        let new_root_pose = Isometry3::<Float>::identity();
        let new_target_pose = abs_pose_map.get(&new_root).expect("outlier_rejection_dual_pairwise: root pose not found").inverse()*abs_pose_map.get(&new_target).expect("outlier_rejection_dual_pairwise: target pose not found");
        let new_abs_pose_map = HashMap::<usize,Isometry3<Float>>::from([(new_root, new_root_pose), (new_target, new_target_pose)]);
        
        // Get all current ids from feature map
        // recompute current abs pose 
        // Remap the ids and keep mapping struct
        let mut feature_map_filtered = feature_map.iter().filter(|(&k,_)| (new_root == k) || (new_target == k)).map(|(k,v)| (*k,v.clone())).collect::<HashMap<_,_>>();
        let landmark_ids_filtered = feature_map_filtered.values().flat_map(|features|{
            features.iter().map(|f|f.get_landmark_id().unwrap())
        }).collect::<HashSet<usize>>();
        let (old_new_id_map, new_unique_landmark_ids) = compute_continuous_landmark_ids_from_unique_landmarks(&landmark_ids_filtered,None);

        for features in feature_map_filtered.values_mut() {
            for i in 0..features.len() {
                let f = &features[i];
                let old_id = f.get_landmark_id().unwrap();
                let new_id = old_new_id_map.get(&old_id).unwrap();
                features[i] = f.copy_with_landmark_id(Some(*new_id));
            }
        }
        
        let new_rejected_landmark_ids = outlier_rejection_dual(&new_camera_ids_root_first, &new_unique_landmark_ids,&new_abs_pose_map, &feature_map_filtered, tol);
        landmark_ids_filtered.clone().into_iter().filter(|&id| new_rejected_landmark_ids.contains(old_new_id_map.get(&id).expect("filter_outliers_by_dual_pairwise: new id not present in old new map"))).collect::<HashSet<_>>()
    }).flatten().collect::<HashSet<_>>();

    if !rejected_landmark_ids.is_empty() {
        filter_by_rejected_landmark_ids(&rejected_landmark_ids, unique_landmark_ids, abs_landmark_map,  match_norm_map,  match_map, landmark_map, &mut feature_map, reprojection_error_map);
    }




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

fn compute_features_per_image_map<Feat: Feature + Clone>(match_map: &HashMap<(usize,usize), Vec<Match<Feat>>>, unique_landmark_ids: &HashSet<usize>, unique_camera_ids: &Vec<usize>) -> HashMap<usize, Vec<Feat>> {
    let mut feature_map = HashMap::<usize, Vec<Feat>>::with_capacity(unique_camera_ids.len());
    for unique_cam_id in unique_camera_ids {
        feature_map.insert(unique_cam_id.clone(),  Vec::<Feat>::with_capacity(unique_landmark_ids.len())); 
    }

    for ((cam_id_1,cam_id_2), matches) in match_map.iter() {

        for vec_idx in 0..matches.len() {
            let m = &matches[vec_idx];
            let f_1 = m.get_feature_one();
            let f_2 = m.get_feature_two();
            feature_map.get_mut(cam_id_1).expect("compute_features_per_image_map: Camera not found in bck feature").push(f_1.clone());
            feature_map.get_mut(cam_id_2).expect("compute_features_per_image_map: Camera not found in bck feature").push(f_2.clone());
        }
    }

    feature_map
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


