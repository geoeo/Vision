extern crate nalgebra as na;

use na::{Vector3, Matrix3, Matrix4, Matrix3xX, DVector, Isometry3};
use std::collections::HashMap;
use crate::image::{
    features::{Feature, Match},
    features::geometry::point::Point,
    triangulation::linear_triangulation
};
use crate::sfm::{bundle_adjustment::state::State, landmark::{Landmark, euclidean_landmark::EuclideanLandmark, inverse_depth_landmark::InverseLandmark}};
use crate::sensors::camera::Camera;
use crate::numerics::pose;
use crate::{Float, reconstruct_original_coordiantes_for_float};

/**
 * For only feature pairs between cams is assumed. Feature triplets etc. are not correctly supported
 */
pub struct CameraFeatureMap {
    pub camera_map: HashMap<u64, (usize, HashMap<usize,usize>)>,
    pub number_of_unique_points: usize,
    /**
     * 2d Vector of rows: point, cols: cam. Where the matrix elements are in (x,y) tuples. 
     * First entry is the all the cams assocaited with a point.
     */
    pub point_cam_map: Vec<Vec<Option<(Float,Float)>>>,
    pub image_row_col: (usize,usize)

}

impl CameraFeatureMap {

    pub const NO_FEATURE_FLAG : Float = -1.0;

    pub fn new<T: Feature>(matches: & Vec<Vec<Match<T>>>, cam_ids: Vec<u64>, image_row_col: (usize,usize)) -> CameraFeatureMap {
        let max_number_of_points = matches.iter().fold(0,|acc,x| acc + x.len());
        let n_cams = cam_ids.len();
        let mut camera_feature_map = CameraFeatureMap{
            camera_map:  HashMap::new(),
            number_of_unique_points: 0,
            point_cam_map: vec![vec![None;n_cams]; max_number_of_points],
            image_row_col
        };

        for i in 0..cam_ids.len(){
            let id = cam_ids[i];
            camera_feature_map.camera_map.insert(id,(i,HashMap::new()));
        }

        camera_feature_map
    }

    fn linear_image_idx(&self, p: &Point<Float>) -> usize {
        (p.y as usize)*self.image_row_col.1+(p.x as usize)
    }


    pub fn add_feature(&mut self, source_cam_id: u64, other_cam_id: u64, 
        x_source: Float, y_source: Float, octave_index_source: usize, 
        x_other: Float, y_other: Float, octave_index_other: usize,  
        pyramid_scale: Float) -> () {

        let (x_source_recon,y_source_recon) = reconstruct_original_coordiantes_for_float(x_source, y_source, pyramid_scale, octave_index_source as i32);
        let (x_other_recon,y_other_recon) = reconstruct_original_coordiantes_for_float(x_other, y_other, pyramid_scale, octave_index_other as i32);
        let point_source = Point::<Float>::new(x_source_recon,y_source_recon);
        let point_other = Point::<Float>::new(x_other_recon,y_other_recon); 
        
        //Linearized Pixel Coordiante as Point ID
        let point_source_idx = self.linear_image_idx(&point_source);
        let point_other_idx = self.linear_image_idx(&point_other);

        let source_cam_idx = self.camera_map.get(&source_cam_id).unwrap().0;
        let other_cam_idx = self.camera_map.get(&other_cam_id).unwrap().0;
        
        let source_point_id =  self.camera_map.get(&source_cam_id).unwrap().1.get(&point_source_idx);
        let other_point_id = self.camera_map.get(&other_cam_id).unwrap().1.get(&point_other_idx);
        
        
        match (source_point_id.clone(),other_point_id.clone()) {
            //If the no point Id is present in either of the two camera it is a new 3D Point
            (None,None) => {
                self.point_cam_map[self.number_of_unique_points][source_cam_idx] = Some((point_source.x,point_source.y));
                self.point_cam_map[self.number_of_unique_points][other_cam_idx] = Some((point_other.x,point_other.y));
                self.camera_map.get_mut(&source_cam_id).unwrap().1.insert(point_source_idx,self.number_of_unique_points);
                self.camera_map.get_mut(&other_cam_id).unwrap().1.insert(point_other_idx, self.number_of_unique_points);

                self.number_of_unique_points += 1;
            },
            // Otherwise add it to the camera which observs it for the first time
            (Some(&point_id),_) => {
                self.point_cam_map[point_id][other_cam_idx] = Some((point_other.x,point_other.y));
                self.camera_map.get_mut(&other_cam_id).unwrap().1.insert(point_other_idx, point_id);

            },
            (None,Some(&point_id)) => {
                self.camera_map.get_mut(&source_cam_id).unwrap().1.insert(point_source_idx,point_id);
                self.point_cam_map[point_id][source_cam_idx] = Some((point_source.x,point_source.y));

            }
        }

    }

    pub fn add_matches<T: Feature>(&mut self, image_id_pairs: &Vec<(u64, u64)>, matches: &Vec<Vec<Match<T>>>, pyramid_scale: Float) -> () {
        assert_eq!(image_id_pairs.len(), matches.len());
        for i in 0..image_id_pairs.len(){
            let (id_a,id_b) = image_id_pairs[i];
            let matches_for_pair = &matches[i];

            for feature_match in matches_for_pair {
                let match_a = &feature_match.feature_one;
                let match_b = &feature_match.feature_two;


                self.add_feature(id_a, id_b, 
                    match_a.get_x_image_float(), match_a.get_y_image_float(),match_a.get_closest_sigma_level(),
                    match_b.get_x_image_float(), match_b.get_y_image_float(),match_b.get_closest_sigma_level(),
                    pyramid_scale);
            }

        }

    }

    /**
     * initial_motion should all be with respect to the first camera
     */
    pub fn get_inverse_depth_landmark_state<C: Camera>(&self, initial_motions : Option<&Vec<(u64,(Vector3<Float>,Matrix3<Float>))>>, inverse_depth_prior: Float, cameras: &Vec<C>) -> State<InverseLandmark,6> {

        let number_of_cameras = self.camera_map.keys().len();
        let number_of_unqiue_landmarks = self.number_of_unique_points;
        let camera_positions = self.get_initial_camera_positions(initial_motions);
        let n_points = self.number_of_unique_points;
        let mut landmarks = Vec::<InverseLandmark>::with_capacity(number_of_unqiue_landmarks);

        for landmark_idx in 0..n_points {
            let observing_cams = &self.point_cam_map[landmark_idx];
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

    /**
     * initial_motion should all be with respect to the first camera
     */
    pub fn get_euclidean_landmark_state<C : Camera + Copy>(&self, initial_motions : Option<&Vec<(u64,(Vector3<Float>,Matrix3<Float>))>>, camera_data: &Vec<((usize, C),(usize,C))>, depth_prior: Float) -> State<EuclideanLandmark,3> {
        
        let number_of_cameras = self.camera_map.keys().len();
        let number_of_unqiue_landmarks = self.number_of_unique_points;


        let landmarks = match initial_motions {
            Some(motions) => {
                assert_eq!(motions.len(), camera_data.len());
                let number_of_camera_pairs = motions.len();
                let mut triangualted_landmarks = vec![EuclideanLandmark::from_state(Vector3::<Float>::zeros()); number_of_unqiue_landmarks];         

                for i in 0..number_of_camera_pairs {
                    let (cam_id,(b,rotation_matrix)) = &motions[i];
                    let ((id_s,camera_matrix_s), (id_f,camera_matrix_f)) = &camera_data[i];

                    let (cam_idx_s, _) = self.camera_map[&(*id_s as u64)];
                    let (cam_idx_f, _) = self.camera_map[&cam_id];
                    
                    assert_eq!(*id_f as u64, *cam_id);

                    let (point_ids, im_s, im_f) = self.get_features_for_cam_pair(cam_idx_s, cam_idx_f);
                    assert_eq!(im_s.len(), im_f.len());
                    let local_landmarks = im_s.len();
                    let local_landmarks_as_float = local_landmarks as Float;
                    let mut normalized_image_points_s = Matrix3xX::<Float>::zeros(local_landmarks);
                    let mut normalized_image_points_f = Matrix3xX::<Float>::zeros(local_landmarks);

                    let mut normalization_matrix_one = Matrix3::<Float>::identity();
                    let mut normalization_matrix_two = Matrix3::<Float>::identity();
                
                    let mut avg_x_one = 0.0;
                    let mut avg_y_one = 0.0;
                    let mut max_dist_one: Float = 0.0;
                    let mut avg_x_two = 0.0;
                    let mut avg_y_two = 0.0;
                    let mut max_dist_two: Float = 0.0;

                    for i in 0..local_landmarks {
                        let (x_s, y_s) = im_s[i];
                        let (x_f, y_f) = im_f[i];
                        let feat_s = Vector3::<Float>::new(x_s,y_s,depth_prior);
                        let feat_f = Vector3::<Float>::new(x_f,y_f,depth_prior);
                        avg_x_one += feat_s[0];
                        avg_y_one += feat_s[1];
                        max_dist_one = max_dist_one.max(feat_s[0].powi(2) + feat_s[1].powi(2));
                        avg_x_two += feat_f[0];
                        avg_y_two += feat_f[1];
                        max_dist_two = max_dist_two.max(feat_f[0].powi(2) + feat_f[1].powi(2));
                        normalized_image_points_s.column_mut(i).copy_from(&feat_s);
                        normalized_image_points_f.column_mut(i).copy_from(&feat_f);
                    }
                    
                    //TODO: unify with five_point and epipolar
                    normalization_matrix_one[(0,2)] = -avg_x_one/local_landmarks_as_float;
                    normalization_matrix_one[(1,2)] = -avg_y_one/local_landmarks_as_float;
                    normalization_matrix_one[(2,2)] = max_dist_one;
                
                    normalization_matrix_two[(0,2)] = -avg_x_two/local_landmarks_as_float;
                    normalization_matrix_two[(1,2)] = -avg_y_two/local_landmarks_as_float;
                    normalization_matrix_two[(2,2)] = max_dist_two;
                
                    normalized_image_points_s = normalization_matrix_one*normalized_image_points_s;
                    normalized_image_points_f = normalization_matrix_two*normalized_image_points_f;

                    let se3 = pose::se3(&b,&rotation_matrix);
                    let projection_1 = camera_matrix_s.get_projection()*(Matrix4::<Float>::identity().fixed_slice::<3,4>(0,0));
                    let projection_2 = camera_matrix_f.get_projection()*(se3.fixed_slice::<3,4>(0,0));
                    
                    //TODO: Push these points into the correct camera coordinate system when using transtitive camera definitions. camera_matrix_s might not be the origin!
                    let triangulated_points = linear_triangulation(&vec!((&normalized_image_points_s,&projection_1),(&normalized_image_points_f,&projection_2)));
                    assert_eq!(triangulated_points.ncols(), point_ids.len());

                    for j in 0..point_ids.len() {
                        let point_id = point_ids[j];
                        let mut point = triangulated_points.fixed_slice::<3, 1>(0, j).into_owned();
                        point /= point[2]*depth_prior;
                        triangualted_landmarks[point_id] = EuclideanLandmark::from_state(point);
                    }

                }
                triangualted_landmarks
            },
            None => {
                vec!(Vector3::<Float>::new(0.0, 0.0, depth_prior);number_of_unqiue_landmarks).iter().map(|&v| EuclideanLandmark::from_state(v)).collect()
            }
        };

        let camera_positions = self.get_initial_camera_positions(initial_motions);
        State::new(camera_positions, landmarks, number_of_cameras, number_of_unqiue_landmarks)
    }

    //Assumes decomposition is relative to reference cam i.e. first cam!
    //TODO: Make this work for transative camera definitions
    fn get_initial_camera_positions(&self,initial_motions : Option<&Vec<(u64,(Vector3<Float>,Matrix3<Float>))>> ) -> DVector::<Float> {

        let number_of_cameras = self.camera_map.keys().len();
        let number_of_cam_parameters = 6*number_of_cameras;
        let mut camera_positions = DVector::<Float>::zeros(number_of_cam_parameters);
        if initial_motions.is_some() {
            let value = initial_motions.unwrap();
            for (cam_id,(h,rotation_matrix)) in value {
                let (cam_idx,_) = self.camera_map[&cam_id];
                let cam_state_idx = 6*cam_idx;
                let rotation = na::Rotation3::from_matrix(&rotation_matrix);
                camera_positions.fixed_slice_mut::<3,1>(cam_state_idx,0).copy_from(&h);
                camera_positions.fixed_slice_mut::<3,1>(cam_state_idx+3,0).copy_from(&rotation.scaled_axis());
            }

        }

        camera_positions

    }

    /**
     * This vector has ordering In the format [f1_cam1, f1_cam2,...] where cam_id(cam_n-1) < cam_id(cam_n) 
     */
    pub fn get_observed_features(&self, invert_feature_y: bool) -> DVector<Float> {
        let n_points = self.number_of_unique_points;
        let n_cams = self.camera_map.keys().len();
        let mut observed_features = DVector::<Float>::zeros(n_points*n_cams*2); // some entries might be invalid
        let c_y = self.image_row_col.0 as Float; 

        for landmark_idx in 0..n_points {
            let observing_cams = &self.point_cam_map[landmark_idx];
            let offset =  2*landmark_idx*n_cams;
            for c in 0..n_cams {
                let feat_id = 2*c + offset;
                let elem = observing_cams[c];
                let (x_val, y_val) = match elem {
                    Some(v) => v,
                    _ => (CameraFeatureMap::NO_FEATURE_FLAG,CameraFeatureMap::NO_FEATURE_FLAG)  
                };
                observed_features[feat_id] = x_val;
                observed_features[feat_id+1] = match invert_feature_y {
                    true => c_y - y_val,
                    false => y_val
                };
            }
        }
        observed_features
    }

    pub fn get_features_for_cam_pair(&self, cam_idx_a: usize, cam_idx_b: usize) -> (Vec<usize>, Vec<(Float,Float)>, Vec<(Float,Float)>) {
        let mut image_coords_a = Vec::<(Float,Float)>::with_capacity(self.number_of_unique_points);
        let mut image_coords_b = Vec::<(Float,Float)>::with_capacity(self.number_of_unique_points);
        let mut point_ids = Vec::<usize>::with_capacity(self.number_of_unique_points);

        for point_idx in 0..self.number_of_unique_points {
            let cam_list = &self.point_cam_map[point_idx];
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