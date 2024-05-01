extern crate nalgebra as na;
extern crate simba;
extern crate num_traits;

use na::{convert,Vector3, Matrix3, DVector, Vector6,Isometry3, Rotation3};
use std::collections::{HashMap,HashSet};
use crate::image::features::{Feature, matches::Match};
use crate::sfm::{state::{State,cam_extrinsic_state::CAMERA_PARAM_SIZE, cam_extrinsic_state::CameraExtrinsicState}, 
    landmark::{Landmark, euclidean_landmark,euclidean_landmark::EuclideanLandmark,inverse_depth_landmark, inverse_depth_landmark::InverseLandmark}};
use crate::sensors::camera::Camera;

use crate::{Float,GenericFloat};

pub const NO_FEATURE_FLAG : Float = -1.0;

pub struct BAStateLinearizer {
    /**
     * This is a map from arbitrary camera ids to linear indices
     */
    pub camera_to_linear_id_map: HashMap<usize, usize>,
    pub landmark_to_linear_id_map: HashMap<usize, usize>
}

impl BAStateLinearizer {
    pub fn new(paths: &Vec<(usize,usize)>, unique_landmark_id_set: &HashSet<usize>) -> BAStateLinearizer {
        let cam_id_set = paths.iter().map(|(v1,v2)| vec![*v1,*v2]).flatten().collect::<HashSet<_>>();

        let mut camera_to_linear_id_map = HashMap::<usize, usize>::with_capacity(cam_id_set.len());
        let mut landmark_to_linear_id_map = HashMap::<usize, usize>::with_capacity(unique_landmark_id_set.len());

        // Map Camera ids to 0-based indices.
        for (i, cam_id) in cam_id_set.iter().enumerate() {
            camera_to_linear_id_map.insert(*cam_id,i);
        }

        for (i, landmark_id) in unique_landmark_id_set.iter().enumerate() {
            landmark_to_linear_id_map.insert(*landmark_id,i);
        }

        BAStateLinearizer {camera_to_linear_id_map, landmark_to_linear_id_map}
    }

    pub fn get_cam_state_idx(&self, cam_id: &usize) -> usize {
        self.camera_to_linear_id_map.get(cam_id).expect("Cam id not present in map").clone()
    }

    pub fn get_inverse_depth_landmark_state<F: GenericFloat, Feat: Feature, C: Camera<Float>>(
        &self, 
        paths: &Vec<(usize,usize)>,
        match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>, 
        abs_pose_map: &HashMap<usize, Isometry3<Float>>,
        reprojection_error_map: &HashMap<(usize, usize),DVector<Float>>,
        camera_norm_map: &HashMap<usize, C>) 
        -> (State<F, impl Landmark<F,{inverse_depth_landmark::LANDMARK_PARAM_SIZE}>,CameraExtrinsicState<F>,{inverse_depth_landmark::LANDMARK_PARAM_SIZE}, CAMERA_PARAM_SIZE>, DVector<F>) { 

            let number_of_cameras = self.camera_to_linear_id_map.len();
            let number_of_unqiue_landmarks = self.landmark_to_linear_id_map.len();

            let mut landmarks = vec![InverseLandmark::from_state(Vector6::<F>::new(F::zero(),F::zero(),F::zero(),F::zero(), F::zero(), F::zero())); number_of_unqiue_landmarks];
            let mut landmark_reprojection_error_map = HashMap::<usize, Float>::with_capacity(number_of_unqiue_landmarks);
    
            let mut feature_location_lookup = vec![vec![None;number_of_cameras]; number_of_unqiue_landmarks];
    
            for (id_s, id_f) in paths {
                let landmark_key = (*id_s, *id_f);
                let matches = match_map.get(&landmark_key).expect("not matches found for path pair");
                let reprojection_errors = reprojection_error_map.get(&landmark_key).expect(format!("no reprojection errors found for key: {:?}",landmark_key).as_str());
                let source_camera = camera_norm_map.get(&id_s).expect("Camera not found");
                let source_camera_pose = abs_pose_map.get(&id_s).expect("Camera Pose not found");
                let internal_source_cam_id = self.camera_to_linear_id_map.get(&id_s).unwrap();
                let internal_other_cam_id = self.camera_to_linear_id_map.get(&id_f).unwrap();
    
                for m_i in 0..matches.len() {
                    let m = &matches[m_i];
                    let point_source_x_float = m.get_feature_one().get_x_image_float();
                    let point_source_y_float = m.get_feature_one().get_y_image_float();
            
                    let point_other_x_float = m.get_feature_two().get_x_image_float();
                    let point_other_y_float = m.get_feature_two().get_y_image_float();
    
                    let landmark_id = &m.get_landmark_id().expect(format!("no landmark id found for match: {:?}",landmark_key).as_str());
                    let linear_landmark_id = self.landmark_to_linear_id_map.get(landmark_id).unwrap().clone();

                    let reprojection_error = reprojection_errors[m_i];

                    // In the Paper the first observed camera position only is saved as state. However, in that paper an EKF was used.
                    // In our case we do a BA, as such we are interested in all feature positions.
                    match landmark_reprojection_error_map.contains_key(landmark_id) {
                        true => {
                            let current_reproj_error =  *landmark_reprojection_error_map.get(&landmark_id).unwrap();
                            if reprojection_error < current_reproj_error {
                                landmark_reprojection_error_map.insert(*landmark_id,reprojection_error);
                                landmarks[linear_landmark_id] = InverseLandmark::new(
                                    &source_camera_pose.cast::<F>(),
                                    m.get_feature_one(),
                                    convert(0.1),
                                    &source_camera.get_inverse_projection(),
                                    &Some(*landmark_id)
                                );
    
                                feature_location_lookup[linear_landmark_id][*internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                                feature_location_lookup[linear_landmark_id][*internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                            }
                        },
                        false => {
                            landmark_reprojection_error_map.insert(*landmark_id,reprojection_error);
                            landmarks[linear_landmark_id] = InverseLandmark::new(
                                &source_camera_pose.cast::<F>(),
                                m.get_feature_one(),
                                convert(0.1),
                                &source_camera.get_inverse_projection(),
                                &Some(*landmark_id)
                            );
    
                            feature_location_lookup[linear_landmark_id][*internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                            feature_location_lookup[linear_landmark_id][*internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                        }
                    }   
                }
            }
    
            let max_depth = landmarks.iter().reduce(|acc, l| {
                if num_traits::float::Float::abs(l.get_state_as_vector().z) > num_traits::float::Float::abs(acc.get_state_as_vector().z) { l } else { acc }
            }).expect("triangulated landmarks empty!").get_state_as_vector().z;
    
            println!("Max depth: {}", max_depth);
    
            let observed_features = Self::get_observed_features::<F>(
                &feature_location_lookup,
                number_of_unqiue_landmarks,
                number_of_cameras
            );
            
            let camera_positions = self.get_initial_camera_positions(abs_pose_map);
            (State::new(camera_positions, landmarks, &self.camera_to_linear_id_map, number_of_cameras, number_of_unqiue_landmarks), observed_features)

    }

    /**
     * @Return: An object holding camera positions and 3d landmarks, 2d Vector of rows: point, cols: cam. Where the matrix elements are in (x,y) tuples. 
     *  First entry in 2d Vector is all the cams assocaited with a point. feature_location_lookup[point_id][cam_id]
     */
    pub fn get_euclidean_landmark_state<F: GenericFloat, Feat: Feature>(
        &self, 
        paths: &Vec<(usize,usize)>,
        match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>, 
        abs_pose_map: &HashMap<usize, Isometry3<Float>>,
        abs_landmark_map: &HashMap<(usize,usize), Vec<EuclideanLandmark<Float>>>,
        reprojection_error_map: &HashMap<(usize, usize),DVector<Float>>) 
        -> (State<F, impl Landmark<F,{euclidean_landmark::LANDMARK_PARAM_SIZE}>,CameraExtrinsicState<F>,{euclidean_landmark::LANDMARK_PARAM_SIZE}, CAMERA_PARAM_SIZE>, DVector<F>) {
        
        let number_of_cameras = self.camera_to_linear_id_map.len();
        let number_of_unqiue_landmarks = self.landmark_to_linear_id_map.len();

        let mut landmarks = vec![EuclideanLandmark::from_state(Vector3::<F>::new(F::zero(),F::zero(),F::one())); number_of_unqiue_landmarks];
        let mut landmark_reprojection_error_map = HashMap::<usize, Float>::with_capacity(number_of_unqiue_landmarks);

        let mut feature_location_lookup = vec![vec![None;number_of_cameras]; number_of_unqiue_landmarks];

        for (id_s, id_f) in paths {
            let landmark_key = (*id_s, *id_f);
            let matches = match_map.get(&landmark_key).expect("not matches found for path pair");
            let reprojection_errors = reprojection_error_map.get(&landmark_key).expect(format!("no reprojection errors found for key: {:?}",landmark_key).as_str());
            let root_aligned_triangulated_matches_s = abs_landmark_map.get(&landmark_key).expect(format!("no landmarks found for key: {:?}",landmark_key).as_str());
            let internal_source_cam_id = self.camera_to_linear_id_map.get(&id_s).unwrap();
            let internal_other_cam_id = self.camera_to_linear_id_map.get(&id_f).unwrap();

            for m_i in 0..matches.len() {
                let m = &matches[m_i];
                let point_source_x_float = m.get_feature_one().get_x_image_float();
                let point_source_y_float = m.get_feature_one().get_y_image_float();
        
                let point_other_x_float = m.get_feature_two().get_x_image_float();
                let point_other_y_float = m.get_feature_two().get_y_image_float();

                let landmark_id = &m.get_landmark_id().expect(format!("no landmark id found for match: {:?}",landmark_key).as_str());
                let linear_landmark_id = self.landmark_to_linear_id_map.get(landmark_id).unwrap().clone();
                let point_s = root_aligned_triangulated_matches_s[m_i].get_euclidean_representation();

                assert_eq!(m.get_landmark_id(),root_aligned_triangulated_matches_s[m_i].get_id());
                
                let reprojection_error = reprojection_errors[m_i];
                match landmark_reprojection_error_map.contains_key(landmark_id) {
                    true => {
                        let current_reproj_error =  *landmark_reprojection_error_map.get(&landmark_id).unwrap();
                        if reprojection_error < current_reproj_error {
                            landmark_reprojection_error_map.insert(*landmark_id,reprojection_error);
                            landmarks[linear_landmark_id] = EuclideanLandmark::from_state_with_id(
                                Vector3::<F>::new(
                                    convert(point_s[0]),
                                    convert(point_s[1]),
                                    convert(point_s[2])
                                ), 
                                &Some(*landmark_id)
                            );

                            feature_location_lookup[linear_landmark_id][*internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                            feature_location_lookup[linear_landmark_id][*internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                        }
                    },
                    false => {
                        landmark_reprojection_error_map.insert(*landmark_id,reprojection_error);
                        landmarks[linear_landmark_id] = EuclideanLandmark::from_state_with_id(
                            Vector3::<F>::new(
                                convert(point_s[0]),
                                convert(point_s[1]),
                                convert(point_s[2])
                            ), 
                            &Some(*landmark_id));

                        feature_location_lookup[linear_landmark_id][*internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                        feature_location_lookup[linear_landmark_id][*internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                    }
                }   



            }
        }

        let max_depth = landmarks.iter().reduce(|acc, l| {
            if num_traits::float::Float::abs(l.get_state_as_vector().z) > num_traits::float::Float::abs(acc.get_state_as_vector().z) { l } else { acc }
        }).expect("triangulated landmarks empty!").get_state_as_vector().z;

        println!("Max depth: {}", max_depth);

        let observed_features = Self::get_observed_features::<F>(
            &feature_location_lookup,
            number_of_unqiue_landmarks,
            number_of_cameras
        );
        
        let camera_positions = self.get_initial_camera_positions(abs_pose_map);
        (State::new(camera_positions, landmarks, &self.camera_to_linear_id_map, number_of_cameras, number_of_unqiue_landmarks), observed_features)
    }

    fn get_initial_camera_positions<F: GenericFloat>(
        &self, pose_map: &HashMap<usize, Isometry3<Float>>) 
        -> DVector::<F> {

        let number_of_cameras = self.camera_to_linear_id_map.keys().len();
        let number_of_cam_parameters = CAMERA_PARAM_SIZE*number_of_cameras; 
        let mut camera_positions = DVector::<F>::zeros(number_of_cam_parameters);
        for (cam_id, cam_idx) in self.camera_to_linear_id_map.iter() {
            let cam_state_idx = CAMERA_PARAM_SIZE*cam_idx;
            let pose = pose_map.get(&cam_id).expect("pose not found for path pair");
            let translation = pose.translation.vector;
            let rotation_matrix = pose.rotation.to_rotation_matrix().matrix().clone();
            let translation_cast: Vector3<F> = translation.cast::<F>();
            let rotation_matrix_cast: Matrix3<F> = rotation_matrix.cast::<F>();
            let rotation = Rotation3::from_matrix_eps(&rotation_matrix_cast, convert(2e-16), 100, Rotation3::identity());
            camera_positions.fixed_view_mut::<3,1>(cam_state_idx,0).copy_from(&translation_cast);
            camera_positions.fixed_view_mut::<3,1>(cam_state_idx+3,0).copy_from(&rotation.scaled_axis());
            
        }
    
        camera_positions
    }

    /**
     * This vector has ordering In the format [f1_cam1, f1_cam2,...] where cam_id(cam_n-1) < cam_id(cam_n) 
     */
    fn get_observed_features<F:GenericFloat>(feature_location_lookup: &Vec<Vec<Option<(Float,Float)>>>, number_of_unique_landmarks: usize, n_cams: usize) -> DVector<F> {
        let mut observed_features = DVector::<F>::zeros(number_of_unique_landmarks*n_cams*2); // some entries might be invalid
        for landmark_idx in 0..number_of_unique_landmarks {
            let observing_cams = &feature_location_lookup[landmark_idx];
            let offset =  2*landmark_idx*n_cams;
            for c in 0..n_cams {
                let feat_id = 2*c + offset;
                let elem = observing_cams[c];
                let (x_val, y_val) = match elem {
                    Some(v) => v,
                    _ => (NO_FEATURE_FLAG,NO_FEATURE_FLAG)  
                };
                observed_features[feat_id] = convert(x_val);
                observed_features[feat_id+1] = convert(y_val);
            }
        }
        observed_features
    }
}