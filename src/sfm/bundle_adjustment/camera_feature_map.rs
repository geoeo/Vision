extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use simba::scalar::{SubsetOf,SupersetOf};
use std::{ops::Mul,convert::From};
use na::{convert,Vector3, Matrix4xX, Matrix3, Matrix4, DVector, Isometry3, Rotation3, SimdRealField, ComplexField,base::Scalar, RealField};
use num_traits::{float,NumAssign};
use std::collections::HashMap;
use crate::image::{
    features::{Feature, Match},
    features::geometry::point::Point
};
use crate::sfm::{bundle_adjustment::state::State, landmark::{Landmark, euclidean_landmark::EuclideanLandmark, inverse_depth_landmark::InverseLandmark}};
use crate::sensors::camera::Camera;
use crate::Float;

/**
 * For only feature pairs between cams is assumed.
 */
pub struct CameraFeatureMap {
    /**
     * The first map is a map of the cameras index by their unique ids.
     * The tuple is the internal cam id and a second map. 
     * The second map is map of which other cams hold the same reference to that 3d point
     */
    pub camera_map: HashMap<usize, (usize, HashMap<usize,usize>)>,
    pub number_of_unique_landmarks: usize,
    /**
     * 2d Vector of rows: point, cols: cam. Where the matrix elements are in (x,y) tuples. 
     * First entry is all the cams assocaited with a point. feature_location_lookup[point_id][cam_id]
     */
    pub feature_location_lookup: Vec<Vec<Option<(Float,Float)>>>,
    /**
     * Map from (internal cam id s, u_s, v_s interal cam id f, u_f, v_f) -> point id
     */
    pub landmark_match_lookup: HashMap<(usize,usize,usize,usize,usize,usize), usize>,
    pub image_row_col: (usize,usize)

}

impl CameraFeatureMap {

    pub const NO_FEATURE_FLAG : Float = -1.0;

    pub fn new<T: Feature>(matches: & Vec<Vec<Vec<Match<T>>>>, cam_ids: Vec<usize>, image_row_col: (usize,usize)) -> CameraFeatureMap {
        let number_of_landmarks = matches.iter().flatten().fold(0,|acc,x| acc + x.len());
        let n_cams = cam_ids.len();
        let mut camera_feature_map = CameraFeatureMap{
            camera_map:  HashMap::new(),
            number_of_unique_landmarks: 0,
            feature_location_lookup: vec![vec![None;n_cams]; number_of_landmarks],
            landmark_match_lookup: HashMap::new(),
            image_row_col
        };

        for i in 0..cam_ids.len(){
            let id = cam_ids[i];
            camera_feature_map.camera_map.insert(id,(i,HashMap::new()));
        }

        camera_feature_map
    }

    fn linear_image_idx(&self, p_x: usize, p_y: usize) -> usize {
        p_y*self.image_row_col.1+p_x
    }


    //TODO: move landmark id generation to track generation! 
    pub fn add_feature<Feat: Feature>(&mut self, source_cam_id: usize, other_cam_id: usize, m: &Match<Feat>) -> () {
        
        let point_source_x = m.feature_one.get_x_image();
        let point_source_y = m.feature_one.get_y_image();

        let point_other_x = m.feature_two.get_x_image();
        let point_other_y = m.feature_two.get_y_image();

        let point_source_x_float = m.feature_one.get_x_image_float();
        let point_source_y_float = m.feature_one.get_y_image_float();

        let point_other_x_float = m.feature_two.get_x_image_float();
        let point_other_y_float = m.feature_two.get_y_image_float();
        
        //Linearized Pixel Coordiante as Point ID
        let point_source_idx = self.linear_image_idx(point_source_x,point_source_y);
        let point_other_idx = self.linear_image_idx(point_other_x,point_other_y);

        let internal_source_cam_id = self.camera_map.get(&source_cam_id).unwrap().0;
        let internal_other_cam_id = self.camera_map.get(&other_cam_id).unwrap().0;
        
        let source_point_id =  self.camera_map.get(&source_cam_id).unwrap().1.get(&point_source_idx);
        let other_point_id = self.camera_map.get(&other_cam_id).unwrap().1.get(&point_other_idx);
        
        let key = (internal_source_cam_id, point_source_x, point_source_y,
        internal_other_cam_id,point_other_x, point_other_y);
        match (source_point_id.clone(),other_point_id.clone()) {
            //If the no point Id is present in either of the two camera it is a new 3D Point
            (None,None) => {
                self.feature_location_lookup[self.number_of_unique_landmarks][internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                self.feature_location_lookup[self.number_of_unique_landmarks][internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                self.camera_map.get_mut(&source_cam_id).unwrap().1.insert(point_source_idx,self.number_of_unique_landmarks);
                self.camera_map.get_mut(&other_cam_id).unwrap().1.insert(point_other_idx, self.number_of_unique_landmarks);
                self.landmark_match_lookup.insert(key, self.number_of_unique_landmarks);
                self.number_of_unique_landmarks += 1;
            },
            // Otherwise add it to the camera which observs it for the first time
            (Some(&point_id),_) => {
                self.feature_location_lookup[point_id][internal_other_cam_id] = Some((point_other_x_float,point_other_y_float));
                self.camera_map.get_mut(&other_cam_id).unwrap().1.insert(point_other_idx, point_id);
                self.landmark_match_lookup.insert(key, point_id);
            },
            (None,Some(&point_id)) => {
                self.camera_map.get_mut(&source_cam_id).unwrap().1.insert(point_source_idx,point_id);
                self.feature_location_lookup[point_id][internal_source_cam_id] = Some((point_source_x_float,point_source_y_float));
                self.landmark_match_lookup.insert(key, point_id);
            }

        }

    }

    pub fn add_matches<T: Feature>(&mut self, path_id_pairs: &Vec<Vec<(usize, usize)>>, matches: &Vec<Vec<Vec<Match<T>>>>) -> () {
        let path_id_pairs_flattened = path_id_pairs.iter().flatten().collect::<Vec<&(usize, usize)>>();        let matches_flattened = matches.iter().flatten().collect::<Vec<&Vec<Match<T>>>>();
        assert_eq!(path_id_pairs_flattened.len(), matches_flattened.len());
        for i in 0..path_id_pairs_flattened.len(){
            let (id_a, id_b) = path_id_pairs_flattened[i];
            let matches_for_pair = matches_flattened[i];

            for feature_match in matches_for_pair {
                self.add_feature(*id_a, *id_b, feature_match);
            }
        }
    }

    /**
     * initial_motion should all be with respect to the first camera
     */
    pub fn get_inverse_depth_landmark_state<C: Camera<Float>>(&self, paths: &Vec<Vec<(usize,usize)>>, pose_map: &HashMap<(usize, usize), Isometry3<Float>>, inverse_depth_prior: Float, cameras: &Vec<C>) -> State<Float,InverseLandmark<Float>,6> {

        let number_of_cameras = self.camera_map.keys().len();
        let number_of_unqiue_landmarks = self.number_of_unique_landmarks;
        let camera_positions = self.get_initial_camera_positions(paths,pose_map);
        let n_points = self.number_of_unique_landmarks;
        let mut landmarks = Vec::<InverseLandmark<Float>>::with_capacity(number_of_unqiue_landmarks);

        for landmark_idx in 0..n_points {
            let observing_cams = &self.feature_location_lookup[landmark_idx];
            let idx_point = observing_cams.iter().enumerate().find(|(_,item)| item.is_some()).expect("get_inverse_depth_landmark_state: No camera for this landmark found! This should not happen");
            let cam_idx = idx_point.0;
            let cam_state_idx = 6*cam_idx;
            let (x_val, y_val) = idx_point.1.unwrap();
            let point = Point::<Float>::new(x_val,y_val);
            let cam_translation = camera_positions.fixed_slice::<3,1>(cam_state_idx,0).into();
            let cam_axis_angle = camera_positions.fixed_slice::<3,1>(cam_state_idx+3,0).into();
            let isometry = Isometry3::new(cam_translation, cam_axis_angle);
            let initial_inverse_landmark = InverseLandmark::new(&isometry,&point,inverse_depth_prior , &cameras[cam_idx]);
 
            landmarks.push(initial_inverse_landmark);
        }
        
        State::new(camera_positions,landmarks, number_of_cameras, number_of_unqiue_landmarks)
    }

    pub fn get_euclidean_landmark_state<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField + Mul<F> + From<F> + RealField + SubsetOf<Float> + SupersetOf<Float>, Feat: Feature>(
        &self, paths: &Vec<Vec<(usize,usize)>>, 
        match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>, 
        pose_map: &HashMap<(usize, usize), Isometry3<Float>>,
        landmark_map: &HashMap<(usize, usize), Matrix4xX<Float>>,
        reprojection_error_map: &HashMap<(usize, usize),DVector<Float>>) 
        -> State<F, EuclideanLandmark<F>,3> {
        
        let number_of_cameras = self.camera_map.keys().len();
        let number_of_unqiue_landmarks = self.number_of_unique_landmarks;

        let mut landmarks = vec![EuclideanLandmark::from_state(Vector3::<F>::new(F::zero(),F::zero(),-F::one())); number_of_unqiue_landmarks];
        let mut landmark_reprojection_error_map = HashMap::<usize, Float>::with_capacity(number_of_unqiue_landmarks);

        for path in paths {
            let mut pose_acc = Matrix4::<Float>::identity();
            for (id_s, id_f) in path {
                let (local_cam_idx_s, _) = self.camera_map[id_s];
                let (local_cam_idx_f, _) = self.camera_map[id_f];
    
                let matches = match_map.get(&(*id_s, *id_f)).expect("not matches found for path pair");
                let landmark_key = (*id_s, *id_f);
                let triangulated_matches = landmark_map.get(&landmark_key).expect(format!("no landmarks found for key: {:?}",landmark_key).as_str());
                let reprojection_errors = reprojection_error_map.get(&landmark_key).expect(format!("no reprojection errors found for key: {:?}",landmark_key).as_str());
                let root_aligned_triangulated_matches = pose_acc*triangulated_matches;
    
                for m_i in 0..matches.len() {
                    let m = &matches[m_i];
                    let feat_s = &m.feature_one;
                    let u_s = feat_s.get_x_image();
                    let v_s = feat_s.get_y_image();
                    let feat_f = &m.feature_two;
                    let u_f = feat_f.get_x_image();
                    let v_f = feat_f.get_y_image();
    
                    let point_id = self.landmark_match_lookup.get(&(local_cam_idx_s,u_s,v_s,local_cam_idx_f,u_f,v_f)).expect("point id not found");
                    let point = root_aligned_triangulated_matches.fixed_slice::<3, 1>(0, m_i).into_owned();
                    
                    let reprojection_error = reprojection_errors[m_i];
                    match landmark_reprojection_error_map.contains_key(point_id) {
                        true => {
                            let current_reproj_error =  *landmark_reprojection_error_map.get(point_id).unwrap();
                            if reprojection_error < current_reproj_error {
                                landmark_reprojection_error_map.insert(*point_id,reprojection_error);
                                landmarks[*point_id] = EuclideanLandmark::from_state(Vector3::<F>::new(
                                    convert(point[0]),
                                    convert(point[1]),
                                    convert(point[2])
                                ));
                            }
                        },
                        false => {
                            landmark_reprojection_error_map.insert(*point_id,reprojection_error);
                            landmarks[*point_id] = EuclideanLandmark::from_state(Vector3::<F>::new(
                                convert(point[0]),
                                convert(point[1]),
                                convert(point[2])
                            ));
                        }
                    }
                }

                let se3 = pose_map.get(&(*id_s, *id_f)).expect("pose not found for path pair").to_matrix();
                pose_acc = pose_acc*se3;
            }
        }

        let max_depth = landmarks.iter().reduce(|acc, l| {
            if float::Float::abs(l.get_state_as_vector().z) > float::Float::abs(acc.get_state_as_vector().z) { l } else { acc }
        }).expect("triangulated landmarks empty!").get_state_as_vector().z;

        println!("Max depth: {}", max_depth);
        
        let camera_positions = self.get_initial_camera_positions(paths,pose_map);
        State::new(camera_positions, landmarks, number_of_cameras, number_of_unqiue_landmarks)
    }

    fn get_initial_camera_positions<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField + Mul<F> + From<F> + RealField + SubsetOf<Float> + SupersetOf<Float>>(
        &self,paths: &Vec<Vec<(usize,usize)>>, pose_map: &HashMap<(usize, usize), Isometry3<Float>>) 
        -> DVector::<F> {

        let number_of_cameras = self.camera_map.keys().len();
        let number_of_cam_parameters = 6*number_of_cameras;
        let mut camera_positions = DVector::<F>::zeros(number_of_cam_parameters);
        for path in paths {
            let mut rot_acc = Matrix3::<F>::identity();
            let mut trans_acc = Vector3::<F>::zeros();
            for (id_s, id_f) in path {
                let (cam_idx,_) = self.camera_map[&id_f];
                let cam_state_idx = 6*cam_idx;
                let pose = pose_map.get(&(*id_s, *id_f)).expect("pose not found for path pair");
                let translation = pose.translation.vector;
                let rotation_matrix = pose.rotation.to_rotation_matrix().matrix().clone();
                let translation_cast: Vector3<F> = translation.cast::<F>();
                let rotation_matrix_cast: Matrix3<F> = rotation_matrix.cast::<F>();
                trans_acc = rot_acc*translation_cast + trans_acc;
                rot_acc = rot_acc*rotation_matrix_cast;
                let rotation = Rotation3::from_matrix_eps(&rot_acc, convert(2e-16), 100, Rotation3::identity());
                camera_positions.fixed_slice_mut::<3,1>(cam_state_idx,0).copy_from(&trans_acc);
                camera_positions.fixed_slice_mut::<3,1>(cam_state_idx+3,0).copy_from(&rotation.scaled_axis());
            }
        }
    
        camera_positions

    }

    /**
     * This vector has ordering In the format [f1_cam1, f1_cam2,...] where cam_id(cam_n-1) < cam_id(cam_n) 
     */
    pub fn get_observed_features<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField + Mul<F> + From<F> + RealField + SubsetOf<Float> + SupersetOf<Float>>(&self, invert_feature_y: bool) -> DVector<F> {
        let n_cams = self.camera_map.keys().len();
        let mut observed_features = DVector::<F>::zeros(self.number_of_unique_landmarks*n_cams*2); // some entries might be invalid
        let c_y = (self.image_row_col.0 - 1) as Float; 

        for landmark_idx in 0..self.number_of_unique_landmarks {
            let observing_cams = &self.feature_location_lookup[landmark_idx];
            let offset =  2*landmark_idx*n_cams;
            for c in 0..n_cams {
                let feat_id = 2*c + offset;
                let elem = observing_cams[c];
                let (x_val, y_val) = match elem {
                    Some(v) => v,
                    _ => (CameraFeatureMap::NO_FEATURE_FLAG,CameraFeatureMap::NO_FEATURE_FLAG)  
                };
                observed_features[feat_id] = convert(x_val);
                observed_features[feat_id+1] = match invert_feature_y {
                    true => convert(c_y - y_val),
                    false => convert(y_val)
                };
            }
        }
        observed_features
    }

    pub fn get_features_for_cam_pair(&self, cam_idx_a: usize, cam_idx_b: usize) -> (Vec<usize>, Vec<(Float,Float)>, Vec<(Float,Float)>) {
        let mut image_coords_a = Vec::<(Float,Float)>::with_capacity(self.number_of_unique_landmarks);
        let mut image_coords_b = Vec::<(Float,Float)>::with_capacity(self.number_of_unique_landmarks);
        let mut point_ids = Vec::<usize>::with_capacity(self.number_of_unique_landmarks);

        for point_idx in 0..self.number_of_unique_landmarks {
            let cam_list = &self.feature_location_lookup[point_idx];
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