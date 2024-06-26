extern crate nalgebra as na;

use crate::image::features::{
    compute_linear_normalization, feature_track::FeatureTrack, matches::Match,
    Feature
};
use crate::sensors::camera::Camera;
use crate::sfm::bundle_adjustment::ba_config::filtering::{compute_reprojection_ranges, compute_reprojection_maps, compute_landmark_maps};
use crate::sfm::state::{State,cam_state::CamState,landmark::{Landmark, euclidean_landmark::EuclideanLandmark}};
use crate::sfm::bundle_adjustment::ba_config::outlier_rejection::{
    calcualte_disparities, calculate_reprojection_errors,
    filter_by_rejected_landmark_ids,
    dual::outlier_rejection_dual
};
use crate::sfm::{
    epipolar::tensor,
    quest,
    triangulation::Triangulation,
    pnp::pnp_config::PnPConfig
};
use crate::image::pyramid::ba::ba_pyramid::BAPyramid;
use crate::Float;
use na::{DVector, Isometry3, Matrix3, Matrix4};
use std::collections::{HashMap, HashSet};

pub mod conversions;
pub mod filtering;
mod outlier_rejection;

/**
 * We assume that the indices between paths and matches are consistent
 */
pub struct BAConfig<C, Feat: Feature> {
    root: usize,
    paths: Vec<Vec<usize>>,
    camera_norm_map: HashMap<usize, C>, 
    match_norm_map: HashMap<(usize, usize), Vec<Match<Feat>>>,
    pose_map: HashMap<(usize, usize), Isometry3<Float>>, // The pose transforms tuple id 2 into the coordiante system of tuple id 1 - 1_P_2
    abs_pose_map: HashMap<usize, Isometry3<Float>>, // World is the root id
    landmark_map: HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>,
    reprojection_error_map: HashMap<(usize, usize), DVector<Float>>
}

impl<C: Camera<Float> + Copy + Clone, Feat: Feature>
    BAConfig<C, Feat>
{
    pub fn new(
        root: usize,
        paths: &Vec<Vec<usize>>,
        pose_map_gt: Option<HashMap<(usize, usize), Isometry3<Float>>>,
        camera_map: HashMap<usize, C>,
        match_map_initial: &HashMap<(usize, usize), Vec<Match<Feat>>>,
        epipolar_alg: tensor::BifocalType,
        triangulation: Triangulation,
        perc_tresh: Float,
        epipolar_thresh: Float,
        image_width: usize,
        image_height: usize
    ) -> BAConfig<C, Feat> {
        let paths_pairs = conversions::compute_path_id_pairs(root, paths);

        //TODO: Compute Image Score for later filtering
        let matches_with_tracks =
            Self::filter_by_max_tracks(&paths_pairs, &match_map_initial);
        assert!(!Self::check_for_duplicate_pixel_entries(
            &matches_with_tracks
        ));

        //TODO: add option to set windows size > 2 ( 2 is implicit at the moment)
        let mut match_map =
            Self::generate_match_map_with_landmark_ids(root, &paths, matches_with_tracks);


        //TODO: Move this out to a prior stage
        let feature_map_per_cam = Self::generate_features_per_cam(&match_map);
        let pyramid_levels = 4;
        let feature_pyramid_map = feature_map_per_cam.iter().map(|(k,v)| (*k,BAPyramid::new(v,pyramid_levels,image_width,image_height))).collect::<HashMap<usize, BAPyramid<Feat>>>();
        let score_map = Self::compute_image_score_map(&feature_pyramid_map);
        for (cam_id, score) in score_map.iter() {
            println!("Score for Cam {} : {}",cam_id,score);
        }


        let (camera_norm_map, mut match_norm_map) =
            Self::normalize_features_and_cameras(&camera_map, &match_map);

        let pose_map = match pose_map_gt.is_some() {
            true => {
                let map = pose_map_gt.unwrap();
                Self::filter_matches_from_pose(
                    &map,
                    &mut match_map,
                    &mut match_norm_map,
                    &camera_norm_map,
                    perc_tresh,
                    epipolar_thresh,
                );
                map
            }
            false => Self::compute_initial_pose_map_and_filter_matches(
                &mut match_map,
                &mut match_norm_map,
                &camera_norm_map,
                perc_tresh,
                epipolar_thresh,
                epipolar_alg,
            ),
        };
        
        //TODO: Maybe remove this completely and the rejection be handled exclusively by filtring 
        let path_pairs = conversions::compute_path_id_pairs(root, &paths);
        let landmark_map = compute_landmark_maps(&path_pairs, &pose_map, &match_norm_map, &camera_norm_map, triangulation);
        let reprojection_error_map =
            compute_reprojection_maps(
                &path_pairs,
                &landmark_map,
                &pose_map,
                &match_norm_map,
                &camera_norm_map
            );

        let (min_reprojection_error_initial, max_reprojection_error_initial) =
            compute_reprojection_ranges(&reprojection_error_map);
        println!(
            "BA Config: Max Reprojection Error : {}, Min Reprojection Error: {}",
            max_reprojection_error_initial, min_reprojection_error_initial
        );

        let abs_pose_map = conversions::compute_absolute_poses_for_root(root, &paths_pairs, &pose_map);
        BAConfig {
            root,
            paths: paths.clone(),
            camera_norm_map,
            match_norm_map,
            abs_pose_map,
            pose_map,
            landmark_map,
            reprojection_error_map
        }
    }

    pub fn root(&self) -> usize {
        self.root
    }
    pub fn paths(&self) -> &Vec<Vec<usize>> {
        &self.paths
    }
    pub fn camera_norm_map(&self) -> &HashMap<usize, C> {
        &self.camera_norm_map
    }
    pub fn match_norm_map(&self) -> &HashMap<(usize, usize), Vec<Match<Feat>>> {
        &self.match_norm_map
    }
    pub fn abs_pose_map(&self) -> &HashMap<usize, Isometry3<Float>> {
        &self.abs_pose_map
    }
    pub fn pose_map(&self) -> &HashMap<(usize, usize), Isometry3<Float>> {
        &self.pose_map
    }
    pub fn reprojection_error_map(&self) -> &HashMap<(usize, usize), DVector<Float>> {
        &self.reprojection_error_map
    }
    pub fn unique_landmark_ids(&self) -> HashSet<usize> {
        self.landmark_map.values().map(|l_vec| l_vec).flatten().map(|l| l.get_id().expect("No id")).collect::<HashSet<_>>()
    }
    pub fn landmark_map(&self) -> &HashMap<(usize,usize), Vec<EuclideanLandmark<Float>>> {
        &self.landmark_map
    }

    pub fn compute_image_score_map(pyramid_map: &HashMap<usize, BAPyramid<Feat>>) -> HashMap<usize, usize>{
        pyramid_map.iter().map(|(k,v)| (*k,v.calculate_score())).collect::<HashMap<usize, usize>>()
    }

    //TODO: Make this generic in Float, since given state may to be of the same precision as config
    pub fn update_state<
        L: Landmark<Float, LANDMARK_PARAM_SIZE> ,CS: CamState<Float,C, CAMERA_PARAM_SIZE> + Copy,const LANDMARK_PARAM_SIZE: usize, const CAMERA_PARAM_SIZE: usize
    >(&mut self, state: &State<Float, C ,L, CS, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>) -> () {
        let camera_positions = state.get_camera_positions();
        let cameras = state.get_cameras();
        let camera_id_map = state.get_camera_id_map();
        let world_landmarks = state.get_landmarks();

        self.update_camera_state(camera_id_map, &camera_positions, &cameras);

        let cam_pair_keys = self.landmark_map().keys().map(|(id1,id2)| (*id1,*id2)).collect::<Vec<_>>();
        for (cam_1,cam_2) in cam_pair_keys {
            self.update_landmark_state(&cam_1,&cam_2,world_landmarks);
        }
    }


    pub fn update_camera_state(&mut self, state_cam_id_map: &HashMap<usize,usize>, state_camera_positions: &Vec<Isometry3<Float>>, state_camera: &Vec<C>) -> () {

        //First we update all absolute poses
        for (cam_id, idx) in state_cam_id_map {
            let new_cam_pos = state_camera_positions[*idx];
            let new_cam_norm = state_camera[*idx];
            self.abs_pose_map.insert(*cam_id, new_cam_pos.clone());
            self.camera_norm_map.insert(*cam_id,new_cam_norm);
        }

        //Then we recalculate relative poses
        let cam_pair_keys = self.pose_map.keys().map(|key| key.clone()).collect::<Vec<_>>();
        for (id1,id2) in cam_pair_keys {
            let pose1 = self.abs_pose_map.get(&id1).expect(format!("Pose id {} not found in update camera state", id1).as_str());
            let pose2 = self.abs_pose_map.get(&id2).expect(format!("Pose id {} not found in update camera state", id2).as_str());
            let pose1_2 = pose1.inverse()*pose2;
            self.pose_map.insert((id1,id2), pose1_2);
        }
    }

    pub fn 
    update_landmark_state<
        L: Landmark<Float, LANDMARK_PARAM_SIZE>,const LANDMARK_PARAM_SIZE: usize, 
        >(&mut self, cam_id_1: &usize, cam_id_2: &usize,  new_world_landmarks: &Vec<L>) -> () {
        let key = (*cam_id_1,*cam_id_2);
        let current_relative_landmarks = self.landmark_map.get(&key).expect("No Landmarks found");
        let pose_world_cam_1 = self.abs_pose_map.get(&cam_id_1).expect("No cam pose");
        let pose_cam_1_world = pose_world_cam_1.inverse();
        let new_relative_landmark_map = new_world_landmarks.iter().filter(|l| l.get_id().is_some()).map(|l|(l.get_id().unwrap(), l.transform_into_other_camera_frame(&pose_cam_1_world))).collect::<HashMap<_,_>>();
        let mut new_relative_landmarks = current_relative_landmarks.clone();
        for l in new_relative_landmarks.iter_mut() {
            if l.get_id().is_some_and(|id| new_relative_landmark_map.contains_key(&id)) {
                let id = l.get_id().unwrap();
                let new_landmark = new_relative_landmark_map.get(&id).unwrap();
                l.set_state(&new_landmark.get_euclidean_representation().coords);
            }
        }


        //sanity check if the landmarks are consistent
        assert!(new_relative_landmarks.iter().zip(current_relative_landmarks.iter()).map(|(new,old)| {
            match (new.get_id(), old.get_id()) {
                (Some(v_new), Some(v_cur)) => v_new == v_cur,
                _ => false
            }
        }).all(|v| v));

        let cam_1 = self.camera_norm_map.get(cam_id_1).expect("Cam id 1 not found");
        let cam_2 = self.camera_norm_map.get(cam_id_2).expect("Cam id 2 not found");
        let ms = self.match_norm_map.get(&key).expect("Matches not found");
        let transform_c2 = self.pose_map.get(&key).expect("No cam pose").inverse();

        let mut new_reprojection_errors = calculate_reprojection_errors(
            &new_relative_landmarks, 
            ms,
            &Matrix4::<Float>::identity().fixed_view::<3, 4>(0, 0).into_owned(),
            cam_1,
            &transform_c2.to_matrix().fixed_view::<3, 4>(0, 0).into_owned(),
            cam_2,
        );

        let current_reprojection_errors = self.reprojection_error_map.get(&key).unwrap();
        let deltas = &new_reprojection_errors - current_reprojection_errors;

        for i in 0..ms.len() {
            let delta = deltas[i];
            if delta > 0.0 {
                new_reprojection_errors[i] = current_reprojection_errors[i];
                new_relative_landmarks[i] = current_relative_landmarks[i];
                assert_eq!(ms[i].get_landmark_id(), new_relative_landmarks[i].get_id());
            }
        }

        self.landmark_map.insert(key, new_relative_landmarks);
        self.reprojection_error_map.insert(key, new_reprojection_errors);

    }

    pub fn generate_pnp_config_from_cam_id(&self, cam_id: usize) -> PnPConfig<C,Feat> {
        let camera_norm = self.camera_norm_map.get(&cam_id).expect("Camera not found in generate_pnp_config_from_cam_id");
        let camera_pose = self.abs_pose_map.get(&cam_id).expect("Camera pose not found in generate_pnp_config_from_cam_id");
        let abs_landmark_map = conversions::generate_abs_landmark_map(self.root,&self.paths,&self.landmark_map,&self.abs_pose_map);
        let pairs_with_cam_id = self.pose_map.keys().filter(|(id1,id2)| *id1 == cam_id || *id2 == cam_id).map(|(id1,id2)| (*id1,*id2)).collect::<Vec<_>>();
        let match_norm_map_for_cam_pairs = pairs_with_cam_id.iter().map(|k| (k, self.match_norm_map.get(k).expect("Feature map could no be found"))).collect::<Vec<_>>();
        let abs_landmark_map_for_cam_pairs = pairs_with_cam_id.iter().map(|k| abs_landmark_map.get(k).expect("Feature map could no be found")).collect::<Vec<_>>();

        let number_of_landmarks = abs_landmark_map_for_cam_pairs.iter().map(|m| m.len()).sum();
        let number_of_matches: usize = match_norm_map_for_cam_pairs.iter().map(|(_,v)| v.len()).sum();

        let mut landmark_map_by_landmark_id = HashMap::<usize,EuclideanLandmark<Float>>::with_capacity(number_of_landmarks);
        let mut feature_norm_map_by_landmark_id = HashMap::<usize, Feat>::with_capacity(number_of_matches);

        for ((k, ms), landmarks) in match_norm_map_for_cam_pairs.iter().zip(abs_landmark_map_for_cam_pairs.iter()) {
            match k {
                (id1, _) if *id1 == cam_id => {
                    for i in 0..ms.len() {
                        let m = &ms[i];
                        let f = m.get_feature_one();
                        let id = m.get_landmark_id().expect("Match with no landmark id!");
                        let l = landmarks[i];

                        assert_eq!(Some(id),l.get_id());

                        feature_norm_map_by_landmark_id.insert(id,f.clone());
                        landmark_map_by_landmark_id.insert(id, EuclideanLandmark::from_state_with_id(l.get_euclidean_representation().coords, &Some(id)));
                    }

                },
                (_, id2) if *id2 == cam_id => {
                    for i in 0..ms.len() {
                        let m = &ms[i];
                        let f = m.get_feature_two();
                        let id = m.get_landmark_id().expect("Match with no landmark id!");
                        let l = landmarks[i];

                        assert_eq!(Some(id),l.get_id());
                        
                        feature_norm_map_by_landmark_id.insert(id,f.clone());
                        landmark_map_by_landmark_id.insert(id, EuclideanLandmark::from_state_with_id(l.get_euclidean_representation().coords, &Some(id)));
                    }
                },
                _ => panic!("Invalid key for pnp generation")
            }

        }

        PnPConfig::new(camera_norm, &landmark_map_by_landmark_id, &feature_norm_map_by_landmark_id, &Some(camera_pose.clone()))
    }

    fn check_for_duplicate_pixel_entries(matches: &Vec<Vec<Vec<Match<Feat>>>>) -> bool {
        let mut duplicates_found = false;
        for path in matches {
            for tracks in path {
                let mut pixel_map =
                    HashMap::<(usize, usize), (Float, Float, Float, Float)>::with_capacity(
                        1000 * 1000,
                    );
                for m in tracks {
                    let f = m.get_feature_one().get_as_2d_point();
                    let f2 = m.get_feature_two().get_as_2d_point();
                    let old_value = pixel_map.insert(
                        (f[0].trunc() as usize, f[1].trunc() as usize),
                        (f[0], f[1], f2[0], f2[1]),
                    );
                    if old_value.is_some() {
                        duplicates_found = true;
                        println!("Warning: duplicate source entries in track: new: (x: {} y: {}, x f: {} y f: {}), old: (x: {} y: {}, x f: {} y f: {})",  f[0], f[1], f2[0], f2[1], old_value.unwrap().0, old_value.unwrap().1,old_value.unwrap().2, old_value.unwrap().3);
                    }
                }
            }
        }
        duplicates_found
    }

    fn generate_match_map_with_landmark_ids(
        root: usize,
        paths: &Vec<Vec<usize>>,
        matches: Vec<Vec<Vec<Match<Feat>>>>,
    ) -> HashMap<(usize, usize), Vec<Match<Feat>>> {
        let number_of_paths = paths.len();
        let match_vec_capacity = 500; //TODO: Set this in a better way
        let mut match_map =
            HashMap::<(usize, usize), Vec<Match<Feat>>>::with_capacity(match_vec_capacity);
        for path_idx in 0..number_of_paths {
            let path = paths[path_idx].clone();
            let match_per_path = matches[path_idx].clone();
            for j in 0..path.len() {
                let id1 = match j {
                    0 => root,
                    idx => path[idx - 1],
                };
                let id2 = path[j];
                let m = &match_per_path[j];
                match_map.insert((id1, id2), m.clone());
            }
        }

        match_map
    }

    fn generate_features_per_cam(feature_map: &HashMap<(usize, usize), Vec<Match<Feat>>>) -> HashMap<usize, Vec<Feat>> {
        let mut target_map = HashMap::<usize,  Vec<Feat>>::new();
        for &(id1,id2) in feature_map.keys() {
            target_map.insert(id1,  Vec::<Feat>::with_capacity(feature_map.get(&(id1,id2)).unwrap().len()));
            target_map.insert(id2,  Vec::<Feat>::with_capacity(feature_map.get(&(id1,id2)).unwrap().len())); 
        }

        for (key,matches) in feature_map.iter() {
            let (id1, id2) = key;
            let mut features_one = Vec::<Feat>::with_capacity(target_map.get(id1).unwrap().len());
            let mut features_two = Vec::<Feat>::with_capacity(target_map.get(id2).unwrap().len());

            for m in matches {
                features_one.push(m.get_feature_one().clone());
                features_two.push(m.get_feature_two().clone());
            }

            target_map.get_mut(id1).unwrap().extend(features_one);
            target_map.get_mut(id2).unwrap().extend(features_two);
        }

        target_map
    }

    fn compute_disparity_map(
        root: usize,
        paths: &Vec<Vec<usize>>,
        match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>,
    ) -> HashMap<(usize, usize), DVector<Float>> {
        let mut disparity_map =
            HashMap::<(usize, usize), DVector<Float>>::with_capacity(match_map.len());
        let path_pairs = conversions::compute_path_id_pairs(root, paths);
        for path in &path_pairs {
            for path_pair in path {
                let ms = match_map.get(path_pair).expect(
                    format!(
                        "compute_disparity_maps: matches not found with key: {:?}",
                        path_pair
                    )
                    .as_str(),
                );
                let disparities = calcualte_disparities(ms);
                disparity_map.insert(*path_pair, disparities);
            }
        }
        disparity_map
    }

    //TODO: merges in tracks originating at the root
    fn filter_by_max_tracks(
        all_paths: &Vec<Vec<(usize, usize)>>,
        match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>,
    ) -> Vec<Vec<Vec<Match<Feat>>>> {
        let mut filtered_matches = Vec::<Vec<Vec<Match<Feat>>>>::with_capacity(all_paths.len());
        let mut feature_tracks = Vec::<Vec<FeatureTrack<Feat>>>::with_capacity(all_paths.len());

        for path_idx in 0..all_paths.len() {
            let path = &all_paths[path_idx];
            let path_len = path.len();
            filtered_matches.push(Vec::<Vec<Match<Feat>>>::with_capacity(path_len));
            feature_tracks.push(Vec::<FeatureTrack<Feat>>::with_capacity(path_len));

            for pair_idx in 0..path_len {
                let k = path[pair_idx];
                filtered_matches[path_idx].push(Vec::<Match<Feat>>::with_capacity(
                    match_map.get(&k).unwrap().len(),
                ));
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
                let mut pixel_set = HashSet::<(usize, usize)>::with_capacity(current_matches.len());
                for m in current_matches {
                    let f1 = m.get_feature_one();
                    let f1_x = f1.get_x_image();
                    let f1_y = f1.get_y_image();
                    let k = (f1_x, f1_y);
                    match img_idx {
                        0 => {
                            if !pixel_set.contains(&k) {
                                let mut id: Option<usize> = None;
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
                                    feature_tracks[path_idx].push(FeatureTrack::new(
                                        max_path_len,
                                        path_idx,
                                        0,
                                        id.unwrap(),
                                        m,
                                    ));
                                } else {
                                    feature_tracks[path_idx].push(FeatureTrack::new(
                                        max_path_len,
                                        path_idx,
                                        0,
                                        landmark_id,
                                        m,
                                    ));
                                    landmark_id += 1;
                                }
                                pixel_set.insert(k);
                            }
                        }
                        img_idx => {
                            let current_feature_one = f1;
                            let mut found_track = false;
                            //TODO: Speed up with caching
                            for track in feature_tracks[path_idx].iter_mut() {
                                if (track.get_current_feature_dest() == current_feature_one.clone())
                                    && (track.get_path_img_id() == (path_idx, img_idx - 1))
                                    && !pixel_set.contains(&k)
                                {
                                    track.add(path_idx, img_idx, m);
                                    found_track = true;
                                    pixel_set.insert(k);
                                    break;
                                }
                            }
                            if !(found_track || pixel_set.contains(&k)) {
                                feature_tracks[path_idx].push(FeatureTrack::new(
                                    max_path_len,
                                    path_idx,
                                    img_idx,
                                    landmark_id,
                                    m,
                                ));
                                landmark_id += 1;
                                pixel_set.insert(k);
                            }
                        }
                    };
                }
            }
        }

        let max_track_score = feature_tracks
            .iter()
            .map(|l| {
                l.iter()
                    .map(|x| x.get_track_score())
                    .reduce(|max, l| if l > max { l } else { max })
                    .expect("filter_by_max_tracks: tracks is empty!")
            })
            .collect::<Vec<usize>>();

        println!("Max Track len: {:?}", max_track_score);

        let max_tracks: Vec<Vec<FeatureTrack<Feat>>> = feature_tracks
            .into_iter()
            .zip(max_track_score)
            .map(|(xs, max)| {
                xs.into_iter()
                    .filter(|x| x.get_track_score() == max)
                    .collect()
            })
            .collect();
        //let max_tracks: Vec<Vec<FeatureTrack<Feat>>> = feature_tracks.into_iter().zip(max_track_lengths).map(| (xs, max) | xs.into_iter().filter(|x| (x.get_track_length() == max) || (x.get_track_length() == max -1)).collect()).collect();
        //let max_tracks = feature_tracks;

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

    fn normalize_features_and_cameras(
        camera_map: &HashMap<usize, C>,
        match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>,
    ) -> (HashMap<usize, C>, HashMap<(usize, usize), Vec<Match<Feat>>>) {
        let mut feature_map = HashMap::<usize, Vec<Feat>>::with_capacity(camera_map.len());
        let mut normalization_map =
            HashMap::<usize, Matrix3<Float>>::with_capacity(camera_map.len());
        let mut camera_norm_map = HashMap::<usize, C>::with_capacity(camera_map.len());
        let mut match_norm_map =
            HashMap::<(usize, usize), Vec<Match<Feat>>>::with_capacity(match_map.len());

        for key in camera_map.keys() {
            //TODO: Better size estimation
            feature_map.insert(*key, Vec::<Feat>::with_capacity(500));
        }

        for ((id1, id2), matches) in match_map.iter() {
            feature_map
                .get_mut(id1)
                .expect("compute_pose_map: camera doesnt exist")
                .extend(matches.iter().map(|m| m.get_feature_one().clone()));
            feature_map
                .get_mut(id2)
                .expect("compute_pose_map: camera doesnt exist")
                .extend(matches.iter().map(|m| m.get_feature_two().clone()));
            match_norm_map.insert(
                (*id1, *id2),
                Vec::<Match<Feat>>::with_capacity(matches.len()),
            );
        }

        for (cam_id, features) in feature_map.iter() {
            let (norm, norm_inv) = compute_linear_normalization(features);
            let c = camera_map
                .get(cam_id)
                .expect("compute_pose_map: cam not found");
            let camera_matrix = norm * c.get_projection();
            let inverse_camera_matrix = c.get_inverse_projection() * norm_inv;
            let c_norm: C = Camera::from_matrices(&camera_matrix, &inverse_camera_matrix);
            normalization_map.insert(*cam_id, norm);
            camera_norm_map.insert(*cam_id, c_norm);
        }

        let match_keys: Vec<_> = match_map.iter().map(|(k, _)| *k).collect();
        for (id1, id2) in match_keys {
            let key = (id1, id2);
            let m = match_map
                .get(&key)
                .expect(format!("match not found with key: {:?}", key).as_str());

            let norm_one = normalization_map
                .get(&id1)
                .expect("compute_pose_map: normalization matrix not found");
            let norm_two = normalization_map
                .get(&id2)
                .expect("compute_pose_map: normalization matrix not found");
            let m_norm = m
                .iter()
                .map(|ma| ma.apply_normalisation(&norm_one, &norm_two, 1.0))
                .collect::<Vec<_>>();
            match_norm_map.insert((id1, id2), m_norm);
        }
        (camera_norm_map, match_norm_map)
    }

    fn filter_matches_from_pose(
        pose_map: &HashMap<(usize, usize), Isometry3<Float>>,
        match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        camera_norm_map: &HashMap<usize, C>,
        perc_tresh: Float,
        epipolar_tresh: Float,
    ) -> () {
        let match_keys: Vec<_> = match_norm_map.iter().map(|(k, _)| *k).collect();
        for (id1, id2) in match_keys {
            let c1 = camera_norm_map
                .get(&id1)
                .expect("compute_pose_map: could not get previous cam");
            let c2 = camera_norm_map
                .get(&id2)
                .expect("compute_pose_map: could not get second camera");
            let key = (id1, id2);
            let m_norm = match_norm_map
                .get(&key)
                .expect(format!("norm match not found with key: {:?}", key).as_str());
            let m = match_map
                .get(&key)
                .expect(format!("match not found with key: {:?}", key).as_str());

            let inverse_camera_matrix_one = c1.get_inverse_projection();
            let inverse_camera_matrix_two = c2.get_inverse_projection();
            let pose = pose_map
                .get(&key)
                .expect("Pose lookup for filter_matches_from_pose failed");

            let e = tensor::essential_matrix_from_motion(
                &pose.translation.vector,
                &pose.rotation.to_rotation_matrix().matrix(),
            );
            let f = tensor::compute_fundamental(
                &e,
                &inverse_camera_matrix_one,
                &inverse_camera_matrix_two,
            );

            let filtered_indices = tensor::select_best_matches_from_fundamental(
                &f,
                m_norm,
                perc_tresh,
                epipolar_tresh,
                1.0,
            );
            let filtered_norm = filtered_indices
                .iter()
                .map(|i| m_norm[*i].clone())
                .collect::<Vec<Match<Feat>>>();
            let filtered = filtered_indices
                .iter()
                .map(|i| m[*i].clone())
                .collect::<Vec<Match<Feat>>>();

            let _ = match_norm_map.insert(key, filtered_norm);
            let _ = match_map.insert(key, filtered);
        }
    }

    #[allow(non_snake_case)]
    fn compute_initial_pose_map_and_filter_matches(
        match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        camera_norm_map: &HashMap<usize, C>,
        perc_tresh: Float,
        epipolar_tresh: Float,
        epipolar_alg: tensor::BifocalType,
    ) -> HashMap<(usize, usize), Isometry3<Float>> {
        let mut pose_map =
            HashMap::<(usize, usize), Isometry3<Float>>::with_capacity(match_norm_map.len());
        let match_keys: Vec<_> = match_norm_map.iter().map(|(k, _)| *k).collect();

        for (id1, id2) in match_keys {
            let c1 = camera_norm_map
                .get(&id1)
                .expect("compute_pose_map: could not get previous cam");
            let c2 = camera_norm_map
                .get(&id2)
                .expect("compute_pose_map: could not get second camera");
            let key = (id1, id2);
            let m_norm = match_norm_map
                .get(&key)
                .expect(format!("norm match not found with key: {:?}", key).as_str());
            let m = match_map
                .get(&key)
                .expect(format!("match not found with key: {:?}", key).as_str());

            let camera_matrix_one = c1.get_projection();
            let camera_matrix_two = c2.get_projection();
            let inverse_camera_matrix_one = c1.get_inverse_projection();
            let inverse_camera_matrix_two = c2.get_inverse_projection();

            let (e, f_m_norm, f_m) = match epipolar_alg {
                tensor::BifocalType::FUNDAMENTAL => {
                    let f = tensor::fundamental::eight_point_hartley(m_norm, 1.0);

                    let filtered_indices = tensor::select_best_matches_from_fundamental(
                        &f,
                        &m_norm,
                        perc_tresh,
                        epipolar_tresh,
                        1.0,
                    );
                    let filtered_norm = filtered_indices
                        .iter()
                        .map(|i| m_norm[*i].clone())
                        .collect::<Vec<Match<Feat>>>();
                    let filtered = filtered_indices
                        .iter()
                        .map(|i| m[*i].clone())
                        .collect::<Vec<Match<Feat>>>();

                    let e =
                        tensor::compute_essential(&f, &camera_matrix_one, &camera_matrix_two);

                    (e, filtered_norm, filtered)
                }
                tensor::BifocalType::ESSENTIAL => {
                    let e = tensor::five_point_essential(
                        m_norm,
                        &camera_matrix_one,
                        &inverse_camera_matrix_one,
                        &camera_matrix_two,
                        &inverse_camera_matrix_two,
                    );
                    let f = tensor::compute_fundamental(
                        &e,
                        &inverse_camera_matrix_one,
                        &inverse_camera_matrix_two,
                    );

                    let filtered_indices = tensor::select_best_matches_from_fundamental(
                        &f,
                        m_norm,
                        perc_tresh,
                        epipolar_tresh,
                        1.0,
                    );
                    let filtered_norm = filtered_indices
                        .iter()
                        .map(|i| m_norm[*i].clone())
                        .collect::<Vec<Match<Feat>>>();
                    let filtered = filtered_indices
                        .iter()
                        .map(|i| m[*i].clone())
                        .collect::<Vec<Match<Feat>>>();

                    (e, filtered_norm, filtered)
                }
                tensor::BifocalType::ESSENTIALRANSAC => {
                    let e = tensor::ransac_five_point_essential(
                        m_norm,
                        &camera_matrix_one,
                        &inverse_camera_matrix_one,
                        &camera_matrix_two,
                        &inverse_camera_matrix_two,
                        1e0,
                        8e4 as usize,
                    );
                    let f = tensor::compute_fundamental(
                        &e,
                        &inverse_camera_matrix_one,
                        &inverse_camera_matrix_two,
                    );

                    let filtered_indices = tensor::select_best_matches_from_fundamental(
                        &f,
                        m_norm,
                        perc_tresh,
                        epipolar_tresh,
                        1.0,
                    );
                    let filtered_norm = filtered_indices
                        .iter()
                        .map(|i| m_norm[*i].clone())
                        .collect::<Vec<Match<Feat>>>();
                    let filtered = filtered_indices
                        .iter()
                        .map(|i| m[*i].clone())
                        .collect::<Vec<Match<Feat>>>();

                    (e, filtered_norm, filtered)
                }
                tensor::BifocalType::QUEST => {
                    let e = quest::quest_ransac(
                        m_norm,
                        &inverse_camera_matrix_one,
                        &inverse_camera_matrix_two,
                        1e-2,
                        1e4 as usize,
                    );
                    let f = tensor::compute_fundamental(
                        &e,
                        &inverse_camera_matrix_one,
                        &inverse_camera_matrix_two,
                    );

                    let filtered_indices = tensor::select_best_matches_from_fundamental(
                        &f,
                        m_norm,
                        perc_tresh,
                        epipolar_tresh,
                        1.0,
                    );
                    let filtered_norm = filtered_indices
                        .iter()
                        .map(|i| m_norm[*i].clone())
                        .collect::<Vec<Match<Feat>>>();
                    let filtered = filtered_indices
                        .iter()
                        .map(|i| m[*i].clone())
                        .collect::<Vec<Match<Feat>>>();

                    (e, filtered_norm, filtered)
                }
            };

            println!("{:?}: Number of matches: {}", key, &f_m_norm.len());
            // The pose transforms id2 into the coordiante system of id1
            let (iso3_opt, _) = tensor::decompose_essential_förstner(
                &e,
                &f_m_norm,
                &inverse_camera_matrix_one,
                &inverse_camera_matrix_two,
            );
            let _ = match iso3_opt {
                Some(isometry) => pose_map.insert(key, isometry),
                None => {
                    println!(
                        "Warning: Decomposition of essential matrix failed for pair ({},{})",
                        id1, id2
                    );
                    pose_map.insert(key, Isometry3::<Float>::identity())
                }
            };
            let _ = match_norm_map.insert(key, f_m_norm);
            let _ = match_map.insert(key, f_m);
        }

        pose_map
    }

    fn filter_outliers_by_dual(
        tol: Float,
        unique_camera_ids_root_first: &Vec<usize>,
        unique_landmark_ids: &mut HashSet<usize>,
        abs_pose_map: &mut HashMap<usize, Isometry3<Float>>,
        match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        match_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
        landmark_map: &mut HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>,
        reprojection_error_map: &mut HashMap<(usize, usize), DVector<Float>>,
        first_landmark_sighting_map: &mut HashMap<usize, usize>
    ) -> () {
        let feature_map = conversions::compute_features_per_image_map(
            &match_norm_map,
            &unique_landmark_ids,
            unique_camera_ids_root_first,
        );
        let rejected_landmark_ids = outlier_rejection_dual(
            unique_camera_ids_root_first,
            unique_landmark_ids,
            abs_pose_map,
            &feature_map,
            tol,
        );
        if !rejected_landmark_ids.is_empty() {
            //TODO: Check if reprojections actually decreased since we have to theoretical guarantee
            filter_by_rejected_landmark_ids(
                &rejected_landmark_ids,
                match_norm_map,
                landmark_map,
                reprojection_error_map
            );
        }
    }
}


