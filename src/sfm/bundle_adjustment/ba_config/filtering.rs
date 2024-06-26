extern crate nalgebra as na;

use crate::image::features::{Feature,matches::Match};
use crate::numerics::pose::{from_matrix, se3};
use crate::sensors::camera::Camera;
use crate::sfm::state::landmark::{Landmark, euclidean_landmark::EuclideanLandmark};
use crate::sfm::bundle_adjustment::ba_config::outlier_rejection::{
    calcualte_disparities, calculate_reprojection_errors,
    compute_continuous_landmark_ids_from_unique_landmarks, filter_by_rejected_landmark_ids,
    reject_landmark_outliers, reject_matches_via_disparity,
    dual::outlier_rejection_dual
};
use crate::sfm::{
    rotation_avg::optimize_rotations_with_rcd,
    triangulation::{triangulate_matches, Triangulation}
};
use crate::sfm::bundle_adjustment::ba_config::BAConfig;
use crate::sfm::bundle_adjustment::ba_config::conversions;
use crate::{float, Float};
use na::{DVector, Isometry3, Matrix3, Matrix4};
use std::collections::{HashMap, HashSet};

pub fn filter_config<C: Camera<Float> + Copy + Clone, Feat: Feature> (
    ba_config: &mut BAConfig<C,Feat>, 
    landmark_cutoff_thresh: Float, 
    run_outlier_detection_pipeline: bool, 
    refine_rotation_via_rcd:bool,
    triangulation: Triangulation) -> BAConfig<C,Feat> {

    let mut landmark_map = ba_config.landmark_map().clone();
    let mut reprojection_error_map = ba_config.reprojection_error_map().clone();
    let mut match_norm_map = ba_config.match_norm_map().clone();
    let mut pose_map = ba_config.pose_map().clone();
    let mut abs_pose_map = ba_config.abs_pose_map().clone();
    let root = ba_config.root();
    let camera_norm_map = ba_config.camera_norm_map().clone();

    let paths = ba_config.paths().clone();

    let paths_pairs = conversions::compute_path_id_pairs(root, &paths);
    let path_id_pairs_flat = paths_pairs.iter().flatten().collect::<Vec<_>>();
    let camera_ids_root_first = conversions::get_sorted_camera_keys(root, &paths);

    let mut rejected_camera_ids = reject_landmark_outliers(
        &mut landmark_map,
        &mut reprojection_error_map,
        &mut match_norm_map,
        landmark_cutoff_thresh,
    );

    assert!(!rejected_camera_ids.contains(&root));

    let (min_reprojection_error_outlier, max_reprojection_error_outlier) =
        compute_reprojection_ranges(&reprojection_error_map);
    println!("BA Config - Filtering: Max Reprojection Error 2): {}, Min Reprojection Error: {}", max_reprojection_error_outlier, min_reprojection_error_outlier);

    let mut unique_landmark_ids = landmark_map.values().map(|l_vec| l_vec).flatten().map(|l| l.get_id().expect("No id")).collect::<HashSet<_>>();

    if run_outlier_detection_pipeline {
        let tol = 5.0
            / camera_norm_map
                .get(&root)
                .expect("Root Cam missing")
                .get_intrinsics()[0]; // rougly 5 pixels //TODO expose this

        filter_outliers_by_dual_pairwise::<C,Feat>(
            tol,
            &path_id_pairs_flat,
            &camera_ids_root_first,
            &mut unique_landmark_ids,
            &mut abs_pose_map,
            &mut match_norm_map,
            &mut landmark_map,
            &mut reprojection_error_map,
        );

        let new_rejected_camera_ids = reject_landmark_outliers(
            &mut landmark_map,
            &mut reprojection_error_map,
            &mut match_norm_map,
            landmark_cutoff_thresh,
        );
        rejected_camera_ids.extend(new_rejected_camera_ids.iter());
        assert!(!rejected_camera_ids.contains(&root));

        let (min_reprojection_error_outlier_dual, max_reprojection_error_outlier_dual) =
            compute_reprojection_ranges(&reprojection_error_map);
        println!("After DUAL Outlier: SFM Config Max Reprojection Error 2): {}, Min Reprojection Error: {}", max_reprojection_error_outlier_dual, min_reprojection_error_outlier_dual);
    }


    let new_pose_map = match refine_rotation_via_rcd {
        true => refine_rotation_by_rcd(root, &paths, &pose_map),
        false => pose_map.clone()
    };

    // We re-triangulate regardless of rcd because matches, landmarks may have been removed by the outlier rejection
    let path_pairs = conversions::compute_path_id_pairs(root, &paths);
    let mut new_landmark_map = compute_landmark_maps(&path_pairs, &new_pose_map, &match_norm_map, &camera_norm_map, triangulation);
    let mut new_reprojection_error_map =
        compute_reprojection_maps(
            &path_pairs,
            &new_landmark_map,
            &new_pose_map,
            &match_norm_map,
            &camera_norm_map
        );

    let keys = landmark_map.keys().map(|k| *k).collect::<Vec<_>>();
    for key in keys {
        let new_reprojection_errors = new_reprojection_error_map.get(&key).unwrap();
        let current_reprojection_errors = reprojection_error_map.get_mut(&key).unwrap();

        if new_reprojection_errors.mean() < current_reprojection_errors.mean() {
            landmark_map.insert(key, new_landmark_map.remove(&key).unwrap());
            reprojection_error_map
                .insert(key, new_reprojection_error_map.remove(&key).unwrap());
            pose_map.insert(key, new_pose_map.get(&key).unwrap().clone());
        }
    }
    abs_pose_map = conversions::compute_absolute_poses_for_root(root, &paths_pairs, &pose_map);
    if run_outlier_detection_pipeline {
        let new_rejected_camera_ids = reject_landmark_outliers(
            &mut landmark_map,
            &mut reprojection_error_map,
            &mut match_norm_map,
            landmark_cutoff_thresh,
        );
        rejected_camera_ids.extend(new_rejected_camera_ids.iter());
        assert!(!rejected_camera_ids.contains(&root));

    }
    let (min_reprojection_error_refined, max_reprojection_error_refined) =
        compute_reprojection_ranges(&reprojection_error_map);
    println!("After Rotation: BA Config Max Reprojection Error 2): {}, Min Reprojection Error: {}", max_reprojection_error_refined, min_reprojection_error_refined);



    for (k,v) in match_norm_map.iter(){
       println!("Final matches for Cam {:?} : {}",k,v.len());
    }

    // for id in &rejected_camera_ids {
    //     println!("Camera id: {} is rejected from paths", id);
    // }

    //@Note This is not enough since a potential transitive pose is not the pose map, etc..
    //let filtered_paths = paths.into_iter().map(|path| path.clone().into_iter().filter(|x| !rejected_camera_ids.contains(&x)).collect::<Vec<usize>>()).collect();


    assert!(rejected_camera_ids.is_empty());

    BAConfig {
        root,
        paths,
        camera_norm_map,
        match_norm_map,
        pose_map,
        abs_pose_map,
        landmark_map,
        reprojection_error_map
    }

}

pub fn compute_reprojection_ranges(
    reprojection_map: &HashMap<(usize, usize), DVector<Float>>,
) -> (Float, Float) {
    let mut max_reprojection_error = float::MIN;
    let mut min_reprojection_error = float::MAX;

    for (_, reprojection_errors) in reprojection_map.iter() {
        max_reprojection_error = max_reprojection_error.max(reprojection_errors.max());
        min_reprojection_error = min_reprojection_error.min(reprojection_errors.min());
    }

    (min_reprojection_error, max_reprojection_error)
}

pub fn refine_rotation_by_rcd(
    root: usize,
    paths: &Vec<Vec<usize>>,
    pose_map: &HashMap<(usize, usize), Isometry3<Float>>,
) -> HashMap<(usize, usize), Isometry3<Float>> {
    let number_of_paths = paths.len();
    let mut new_pose_map =
        HashMap::<(usize, usize), Isometry3<Float>>::with_capacity(pose_map.capacity());
    let mut initial_cam_motions_per_path =
        Vec::<Vec<((usize, usize), Matrix3<Float>)>>::with_capacity(number_of_paths);
    for path_idx in 0..number_of_paths {
        let path = &paths[path_idx];
        let path_length = path.len();
        let mut cam_motions =
            Vec::<((usize, usize), Matrix3<Float>)>::with_capacity(path_length);

        for j in 0..path_length {
            let id1 = match j {
                0 => root,
                idx => path[idx - 1],
            };
            let id2 = path[j];
            let key = (id1, id2);
            let initial_pose = pose_map
                .get(&key)
                .expect("No pose found for rcd rotation in pose map");
            cam_motions.push((
                key,
                initial_pose
                    .rotation
                    .to_rotation_matrix()
                    .matrix()
                    .to_owned(),
            ));
        }
        initial_cam_motions_per_path.push(cam_motions);
    }
    let initial_cam_rotations_per_path_rcd =
        optimize_rotations_with_rcd(&initial_cam_motions_per_path);
    for i in 0..initial_cam_rotations_per_path_rcd.len() {
        let path_len = initial_cam_rotations_per_path_rcd[i].len();
        for j in 0..path_len {
            let (key, rcd_rot) = initial_cam_rotations_per_path_rcd[i][j];
            let initial_pose = pose_map
                .get(&key)
                .expect("No pose found for rcd rotation in pose map");

            //let initial_rot = initial_pose.rotation.to_rotation_matrix().matrix().to_owned();
            //let angular_distance_initial = angular_distance(&initial_rot);
            //let angular_distance_rcd = angular_distance(&rcd_rot);
            //println!("initial r : {}", initial_rot);
            //println!("rcd r : {}",rcd_rot);
            //println!("initial ang dist : {}", angular_distance_initial);
            //println!("rcd ang dist : {}", angular_distance_rcd);

            let new_se3 = se3(&initial_pose.translation.vector, &rcd_rot);
            new_pose_map.insert(key, from_matrix(&new_se3));
        }
    }

    new_pose_map
}

pub fn compute_landmark_maps<C: Camera<Float> + Clone, Feat: Feature>(
    path_pairs: &Vec<Vec<(usize,usize)>>,
    pose_map: &HashMap<(usize,usize), Isometry3<Float>>,
    match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>,
    camera_map: &HashMap<usize, C>,
    triangulation: Triangulation,
) -> HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>> {
    let mut landmark_map =
        HashMap::<(usize, usize), Vec<EuclideanLandmark<Float>>>::with_capacity(match_map.len());


    for path in path_pairs {
        for path_pair in path {
            let root = path_pair.0;
            let sub_path = vec!(vec!(*path_pair));
            let abs_pose_map = conversions::compute_absolute_poses_for_root(root, &sub_path, &pose_map);

            //TODO: need to ensure that the matches are the intersection of all triangulated views. Maybe even do this before when matches are built!
            
            let landmarks = triangulate_matches(
                vec!(*path_pair),
                &abs_pose_map,
                match_map,
                &camera_map,
                triangulation,
            );



            landmark_map.insert(*path_pair, landmarks);
        }
    }

    landmark_map
}

pub fn compute_reprojection_maps<C: Camera<Float> + Clone, Feat: Feature>(
    path_pairs: &Vec<Vec<(usize,usize)>>,
    landmark_map: &HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>,
    pose_map: &HashMap<(usize, usize), Isometry3<Float>>,
    match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>,
    camera_map: &HashMap<usize, C>
) -> HashMap<(usize, usize), DVector<Float>> {
    let mut reprojection_map =
        HashMap::<(usize, usize), DVector<Float>>::with_capacity(match_map.len());

    for path in path_pairs {
        for path_pair in path {

            let landmarks = landmark_map.get(path_pair).expect(format!("compute_reprojection_maps: landmarks not found with key: ({:?})",path_pair).as_str());
            let pose = pose_map.get(path_pair).expect(format!("compute_reprojection_maps: pose not found with key: ({:?})",path_pair).as_str()).to_matrix();
            let ms = match_map.get(path_pair).expect(
                format!(
                    "compute_landmarks_and_reprojection_maps: matches not found with key: {:?}",
                    path_pair
                )
                .as_str(),
            );
            let cam_1 = camera_map
                .get(&path_pair.0)
                .expect("compute_landmarks_and_reprojection_maps: camera 1 not found");
            let cam_2 = camera_map
                .get(&path_pair.1)
                .expect("compute_landmarks_and_reprojection_maps: camera 2 not found");
            let transform_c1 = Matrix4::<Float>::identity()
                .fixed_view::<3, 4>(0, 0)
                .into_owned();
            let transform_c2 = pose.fixed_view::<3, 4>(0, 0).into_owned();
            let reprojection_errors = calculate_reprojection_errors(
                &landmarks,
                ms,
                &transform_c1,
                cam_1,
                &transform_c2,
                cam_2,
            );

            reprojection_map.insert(*path_pair, reprojection_errors);
        }
    }

    reprojection_map
}

pub fn filter_outliers_by_dual_pairwise<C: Camera<Float> + Clone, Feat: Feature>(
    tol: Float,
    path_pairs: &Vec<&(usize, usize)>,
    unique_camera_ids: &Vec<usize>,
    unique_landmark_ids: &mut HashSet<usize>,
    abs_pose_map: &HashMap<usize, Isometry3<Float>>,
    match_norm_map: &mut HashMap<(usize, usize), Vec<Match<Feat>>>,
    landmark_map: &mut HashMap<(usize, usize), Vec<EuclideanLandmark<Float>>>,
    reprojection_error_map: &mut HashMap<(usize, usize), DVector<Float>>
) -> () {
    let feature_map = conversions::compute_features_per_image_map(
        &match_norm_map,
        &unique_landmark_ids,
        unique_camera_ids,
    );

    let rejected_landmark_ids = path_pairs
        .clone()
        .into_iter()
        .map(|&(new_root, new_target)| {
            let new_camera_ids_root_first = vec![new_root, new_target];
            let new_root_pose = Isometry3::<Float>::identity();
            let new_target_pose = abs_pose_map
                .get(&new_root)
                .expect("outlier_rejection_dual_pairwise: root pose not found")
                .inverse()
                * abs_pose_map
                    .get(&new_target)
                    .expect("outlier_rejection_dual_pairwise: target pose not found");
            let new_abs_pose_map = HashMap::<usize, Isometry3<Float>>::from([
                (new_root, new_root_pose),
                (new_target, new_target_pose),
            ]);

            // Get all current ids from feature map
            // recompute current abs pose
            // Remap the ids and keep mapping struct
            let mut feature_map_filtered = feature_map
                .iter()
                .filter(|(&k, _)| (new_root == k) || (new_target == k))
                .map(|(k, v)| (*k, v.clone()))
                .collect::<HashMap<_, _>>();
            let landmark_ids_filtered = feature_map_filtered
                .values()
                .flat_map(|features| features.iter().map(|f| f.get_landmark_id().unwrap()))
                .collect::<HashSet<usize>>();
            let (old_new_id_map, new_unique_landmark_ids) =
                compute_continuous_landmark_ids_from_unique_landmarks(
                    &landmark_ids_filtered,
                    None,
                );

            for features in feature_map_filtered.values_mut() {
                for i in 0..features.len() {
                    let f = &features[i];
                    let old_id = f.get_landmark_id().unwrap();
                    let new_id = old_new_id_map.get(&old_id).unwrap();
                    features[i] = f.copy_with_landmark_id(Some(*new_id));
                }
            }

            // Outlier rejection scheme needs continuous ids
            let new_rejected_landmark_ids = outlier_rejection_dual(
                &new_camera_ids_root_first,
                &new_unique_landmark_ids,
                &new_abs_pose_map,
                &feature_map_filtered,
                tol,
            );

            //Here we map the new ids back to the old ones
            landmark_ids_filtered
                .clone()
                .into_iter()
                .filter(|&id| {
                    new_rejected_landmark_ids.contains(old_new_id_map.get(&id).expect(
                        "filter_outliers_by_dual_pairwise: new id not present in old new map",
                    ))
                })
                .collect::<HashSet<_>>()
        })
        .flatten()
        .collect::<HashSet<_>>();

    if !rejected_landmark_ids.is_empty() {
        filter_by_rejected_landmark_ids(
            &rejected_landmark_ids,
            match_norm_map,
            landmark_map,
            reprojection_error_map
        );
    }
}
