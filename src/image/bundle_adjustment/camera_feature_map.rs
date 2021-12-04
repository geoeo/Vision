extern crate nalgebra as na;

use na::{Vector3,Matrix3,DVector};
use std::collections::HashMap;
use crate::image::{
    features::{Feature,Match},
    features::geometry::point::Point,
    bundle_adjustment::state::State
};
use crate::{Float, reconstruct_original_coordiantes_for_float};



pub struct CameraFeatureMap {
    pub camera_map: HashMap<u64, Vec<(usize,u64)>>,
    /**
     * The image coordiantes in this list have been scaled to the original image resolution
     * For two features f1, f2 the cam id(f1) < cam_id(f2)
     * */ 
    pub feature_list: Vec<(Point<Float>,Point<Float>)>

}

impl CameraFeatureMap {

    pub fn new<T: Feature>(matches: & Vec<Vec<Match<T>>>) -> CameraFeatureMap {
        let number_of_unique_features = matches.iter().fold(0,|acc,x| acc + x.len());
        CameraFeatureMap{
            camera_map: HashMap::new(),
            feature_list: Vec::<(Point<Float>,Point<Float>)>::with_capacity(number_of_unique_features)
        }
        
    }

    pub fn add_images_from_params(&mut self, id: u64, features_per_octave: usize, octave_count: usize) -> () {
        self.add_image(id, features_per_octave, octave_count);
    }

    pub fn add_image(&mut self,id: u64, features_per_octave: usize, octave_count: usize) -> () {
        self.camera_map.insert(id,Vec::<(usize,u64)>::with_capacity(features_per_octave*octave_count));

    }

    pub fn add_feature(&mut self, source_cam_id: u64, other_cam_id: u64, 
        x_source: Float, y_source: Float, octave_index_source: usize, 
        x_other: Float, y_other: Float, octave_index_other: usize,  
        pyramid_scale: Float) -> () {

        let (x_source_recon,y_source_recon) = reconstruct_original_coordiantes_for_float(x_source, y_source, pyramid_scale, octave_index_source as i32);
        let (x_other_recon,y_other_recon) = reconstruct_original_coordiantes_for_float(x_other, y_other, pyramid_scale, octave_index_other as i32);
        let point_source = Point::<Float>::new(x_source_recon,y_source_recon);
        let point_other = Point::<Float>::new(x_other_recon,y_other_recon); 
 
        let idx_in_feature_list = self.feature_list.len();

        match (source_cam_id, other_cam_id){
            (s,o) if s < o => self.feature_list.push((point_source,point_other)),
            _ => self.feature_list.push((point_other,point_source))
        };

        let source_cam_map = self.camera_map.get_mut(&source_cam_id).expect(&format!("No image with id: {} found in map",source_cam_id).to_string());
        source_cam_map.push((idx_in_feature_list,other_cam_id));
        let other_cam_map = self.camera_map.get_mut(&other_cam_id).expect(&format!("No image with id: {} found in map",other_cam_id).to_string());
        other_cam_map.push((idx_in_feature_list,source_cam_id));
    }

    pub fn add_matches<T: Feature>(&mut self, image_id_pairs: &Vec<(u64, u64)>, matches: & Vec<Vec<Match<T>>>, pyramid_scale: Float) -> () {
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
    pub fn get_initial_state(&self, initial_motions : Option<&Vec<(Vector3<Float>,Matrix3<Float>)>>, initial_depth: Float) -> State {

        let number_of_cameras = self.camera_map.keys().len();
        let number_of_unqiue_points = self.feature_list.len();
        let number_of_cam_parameters = 6*number_of_cameras;
        let number_of_point_parameters = 3*number_of_unqiue_points;
        let total_parameters = number_of_cam_parameters+number_of_point_parameters;
        let mut data = DVector::<Float>::zeros(total_parameters);
        
        if initial_motions.is_some() {
            let value = initial_motions.unwrap();
            assert_eq!(value.len(),number_of_cameras/2);
            let mut counter = 0;
            for i in (6..number_of_cam_parameters).step_by(12){
                let idx = match i {
                    v if v == 0 => 0,
                    _ => i/6-1-counter
                };
                let (h,rotation_matrix) = value[idx];
                let rotation = na::Rotation3::from_matrix(&rotation_matrix);
                let rotation_transpose = rotation.transpose();
                let translation = rotation_transpose*(-h);
                data.fixed_slice_mut::<3,1>(i,0).copy_from(&translation);
                data.fixed_slice_mut::<3,1>(i,0).copy_from(&rotation_transpose.scaled_axis());
                counter += 1;
            }

        }

        // Initialise points to a depth of -1
        for i in (number_of_cam_parameters..total_parameters).step_by(3){
            data[i] = 0.0;
            data[i+1] = 0.0;
            data[i+2] = initial_depth; 
        }

        State{data , n_cams: number_of_cameras, n_points: number_of_unqiue_points}
    }

    /**
     * This vector has ordering In the format [f1_cam1, f1_cam2,...] where cam_id(cam_n-1) < cam_id(cam_n) 
     */

    pub fn get_observed_features(&self) -> DVector<Float> {
        let n_points = self.feature_list.len();
        let n_cams = self.camera_map.keys().len();
        let mut observed_features = DVector::<Float>::zeros(n_points*n_cams*2); // some entries might be zero
        let mut sorted_keys = self.camera_map.keys().cloned().collect::<Vec<u64>>();
        sorted_keys.sort_unstable();
        
        //TODO: rewrite this -> very hard to understand logic
        for i in 0..sorted_keys.len() {
            let key = sorted_keys[i];
            let camera_map = self.camera_map.get(&key); 

            let feature_indices = camera_map.expect(&format!("No image with id: {} found in map",key).to_string());
            for j in 0..feature_indices.len() {
                let (idx, other_cam_id) = feature_indices[j];
                let feature_pos = match (key, other_cam_id) {
                    (s,o) if s < o => self.feature_list[idx].0,
                    _ =>  self.feature_list[idx].1
                };
                let feat_id = 2*i + 2*j*n_cams;
                observed_features[feat_id] = feature_pos.x as Float;
                observed_features[feat_id+1] = feature_pos.y as Float;
            }
        }

        observed_features
    }




}