extern crate nalgebra as na;
extern crate num_traits;

use na::{convert,Vector3, Matrix4xX, Matrix3, DVector, Isometry3, Rotation3,base::Scalar, RealField};
use num_traits::{float,NumAssign};
use simba::scalar::SupersetOf;
use std::collections::HashMap;
use crate::image::features::{Feature, matches::Match};
use crate::sfm::{state::State, landmark::{Landmark, euclidean_landmark::EuclideanLandmark}};
use crate::Float;

pub const NO_FEATURE_FLAG : Float = -1.0;

pub struct StateLinearizer {
    /**
     * This is a map from arbitrary camera ids to linear indices
     */
    pub camera_to_linear_id_map: HashMap<usize, usize>,
}

impl StateLinearizer {
    pub fn new(cam_ids: Vec<usize>) -> StateLinearizer {
        let mut camera_to_linear_id_map = HashMap::<usize, usize>::new();
        // Map Camera ids to 0-based indices.
        for i in 0..cam_ids.len(){
            let id = cam_ids[i];
            camera_to_linear_id_map.insert(id,i);
        }

        StateLinearizer {camera_to_linear_id_map}
    }

    /**
     * initial_motion should all be with respect to the first camera
     * //TODO: Rework
     */
    // pub fn get_inverse_depth_landmark_state<C: Camera<Float>>(&self, paths: &Vec<Vec<(usize,usize)>>, abs_pose_map: &HashMap<usize, Isometry3<Float>>, inverse_depth_prior: Float, cameras: &Vec<C>) -> State<Float,InverseLandmark<Float>,6> {
    //     panic!("Rework get_inverse_depth_landmark_state");
    //     // let number_of_cameras = self.camera_map.keys().len();
    //     // let number_of_unqiue_landmarks = self.number_of_unique_landmarks;
    //     // let camera_positions = self.get_initial_camera_positions(paths,abs_pose_map);
    //     // let n_points = self.number_of_unique_landmarks;
    //     // let mut landmarks = Vec::<InverseLandmark<Float>>::with_capacity(number_of_unqiue_landmarks);

    //     // for landmark_idx in 0..n_points {
    //     //     let observing_cams = &self.feature_location_lookup[landmark_idx];
    //     //     let idx_point = observing_cams.iter().enumerate().find(|(_,item)| item.is_some()).expect("get_inverse_depth_landmark_state: No camera for this landmark found! This should not happen");
    //     //     let cam_idx = idx_point.0;
    //     //     let cam_state_idx = 6*cam_idx;
    //     //     let (x_val, y_val) = idx_point.1.unwrap();
    //     //     let point = Point::<Float>::new(x_val,y_val);
    //     //     let cam_translation = camera_positions.fixed_view::<3,1>(cam_state_idx,0).into();
    //     //     let cam_axis_angle = camera_positions.fixed_view::<3,1>(cam_state_idx+3,0).into();
    //     //     let isometry = Isometry3::new(cam_translation, cam_axis_angle);
    //     //     let initial_inverse_landmark = InverseLandmark::new(&isometry,&point,inverse_depth_prior , &cameras[cam_idx]);
 
    //     //     landmarks.push(initial_inverse_landmark);
    //     // }
        
    //     // State::new(camera_positions,landmarks, number_of_cameras, number_of_unqiue_landmarks)
    // }

    /**
     * @Return: An object holding camera positions and 3d landmarks, 2d Vector of rows: point, cols: cam. Where the matrix elements are in (x,y) tuples. 
     *  First entry in 2d Vector is all the cams assocaited with a point. feature_location_lookup[point_id][cam_id]
     */
    pub fn get_euclidean_landmark_state<F: float::Float + Scalar + RealField + SupersetOf<Float>, Feat: Feature>(
        &self, 
        paths: &Vec<Vec<(usize,usize)>>, 
        match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>, 
        abs_pose_map: &HashMap<usize, Isometry3<Float>>,
        abs_landmark_map: &HashMap<usize, Matrix4xX<Float>>,
        reprojection_error_map: &HashMap<(usize, usize),DVector<Float>>,
        number_of_unqiue_landmarks: usize) 
        -> (State<F, EuclideanLandmark<F>,3>, Vec<Vec<Option<(Float,Float)>>>) {
        
        let number_of_cameras = self.camera_to_linear_id_map.keys().len();

        let mut landmarks = vec![EuclideanLandmark::from_state(Vector3::<F>::new(F::zero(),F::zero(),-F::one())); number_of_unqiue_landmarks];
        let mut landmark_reprojection_error_map = HashMap::<usize, Float>::with_capacity(number_of_unqiue_landmarks);

        let mut feature_location_lookup = vec![vec![None;number_of_cameras]; number_of_unqiue_landmarks];

        for path in paths {
            for (id_s, id_f) in path {
                let matches = match_map.get(&(*id_s, *id_f)).expect("not matches found for path pair");
                let landmark_key = (*id_s, *id_f);
                let reprojection_errors = reprojection_error_map.get(&landmark_key).expect(format!("no reprojection errors found for key: {:?}",landmark_key).as_str());
                let root_aligned_triangulated_matches = abs_landmark_map.get(&id_s).expect(format!("no landmarks found for key: {:?}",landmark_key).as_str());
                let internal_source_cam_id = self.camera_to_linear_id_map.get(id_s).unwrap();
                let internal_other_cam_id = self.camera_to_linear_id_map.get(id_f).unwrap();
    
                for m_i in 0..matches.len() {
                    let m = &matches[m_i];
                    let point_source_x_float = m.get_feature_one().get_x_image_float();
                    let point_source_y_float = m.get_feature_one().get_y_image_float();
            
                    let point_other_x_float = m.get_feature_two().get_x_image_float();
                    let point_other_y_float = m.get_feature_two().get_y_image_float();

                    let landmark_id = &m.get_landmark_id().expect(format!("no landmark id found for match: {:?}",landmark_key).as_str());
                    let point = root_aligned_triangulated_matches.fixed_view::<3, 1>(0, m_i).into_owned();
                    
                    let reprojection_error = reprojection_errors[m_i];
                    match landmark_reprojection_error_map.contains_key(landmark_id) {
                        true => {
                            let current_reproj_error =  *landmark_reprojection_error_map.get(&landmark_id).unwrap();
                            if reprojection_error < current_reproj_error {
                                landmark_reprojection_error_map.insert(*landmark_id,reprojection_error);
                                landmarks[*landmark_id] = EuclideanLandmark::from_state(Vector3::<F>::new(
                                    convert(point[0]),
                                    convert(point[1]),
                                    convert(point[2])
                                ));

                                feature_location_lookup[*landmark_id][*internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                                feature_location_lookup[*landmark_id][*internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                            }
                        },
                        false => {
                            landmark_reprojection_error_map.insert(*landmark_id,reprojection_error);
                            landmarks[*landmark_id] = EuclideanLandmark::from_state(Vector3::<F>::new(
                                convert(point[0]),
                                convert(point[1]),
                                convert(point[2])
                            ));

                            feature_location_lookup[*landmark_id][*internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                            feature_location_lookup[*landmark_id][*internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                        }
                    }
                }
            }
        }

        let max_depth = landmarks.iter().reduce(|acc, l| {
            if float::Float::abs(l.get_state_as_vector().z) > float::Float::abs(acc.get_state_as_vector().z) { l } else { acc }
        }).expect("triangulated landmarks empty!").get_state_as_vector().z;

        println!("Max depth: {}", max_depth);
        
        let camera_positions = self.get_initial_camera_positions(paths,abs_pose_map);
        (State::new(camera_positions, landmarks, number_of_cameras, number_of_unqiue_landmarks), feature_location_lookup)
    }

    fn get_initial_camera_positions<F: float::Float + Scalar + RealField + SupersetOf<Float>>(
        &self,paths: &Vec<Vec<(usize,usize)>>, pose_map: &HashMap<usize, Isometry3<Float>>) 
        -> DVector::<F> {

        let number_of_cameras = self.camera_to_linear_id_map.keys().len();
        let number_of_cam_parameters = 6*number_of_cameras;
        let mut camera_positions = DVector::<F>::zeros(number_of_cam_parameters);
        for path in paths {
            for (_, id_f) in path {
                let cam_idx = self.camera_to_linear_id_map[&id_f];
                let cam_state_idx = 6*cam_idx;
                let pose = pose_map.get(&id_f).expect("pose not found for path pair");
                let translation = pose.translation.vector;
                let rotation_matrix = pose.rotation.to_rotation_matrix().matrix().clone();
                let translation_cast: Vector3<F> = translation.cast::<F>();
                let rotation_matrix_cast: Matrix3<F> = rotation_matrix.cast::<F>();
                let rotation = Rotation3::from_matrix_eps(&rotation_matrix_cast, convert(2e-16), 100, Rotation3::identity());
                camera_positions.fixed_view_mut::<3,1>(cam_state_idx,0).copy_from(&translation_cast);
                camera_positions.fixed_view_mut::<3,1>(cam_state_idx+3,0).copy_from(&rotation.scaled_axis());
            }
        }
    
        camera_positions

    }

    /**
     * This vector has ordering In the format [f1_cam1, f1_cam2,...] where cam_id(cam_n-1) < cam_id(cam_n) 
     */
    pub fn get_observed_features<F: float::Float + Scalar + NumAssign + SupersetOf<Float>>(&self, feature_location_lookup: &Vec<Vec<Option<(Float,Float)>>>, number_of_unique_landmarks: usize) -> DVector<F> {
        let n_cams = self.camera_to_linear_id_map.keys().len();
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

    pub fn get_features_for_cam_pair(&self, cam_idx_a: usize, cam_idx_b: usize,  feature_location_lookup: &Vec<Vec<Option<(Float,Float)>>>, number_of_unique_landmarks: usize) 
        -> (Vec<usize>, Vec<(Float,Float)>, Vec<(Float,Float)>) {
        let mut image_coords_a = Vec::<(Float,Float)>::with_capacity(number_of_unique_landmarks);
        let mut image_coords_b = Vec::<(Float,Float)>::with_capacity(number_of_unique_landmarks);
        let mut point_ids = Vec::<usize>::with_capacity(number_of_unique_landmarks);

        for point_idx in 0..number_of_unique_landmarks {
            let cam_list = &feature_location_lookup[point_idx];
            let im_a = cam_list[cam_idx_a];
            let im_b = cam_list[cam_idx_b];

            if im_a.is_some() && im_b.is_some() {
                image_coords_a.push(im_a.unwrap());
                image_coords_b.push(im_b.unwrap());
                point_ids.push(point_idx);
            }
        }

        (point_ids, image_coords_a,image_coords_b)
    }
}