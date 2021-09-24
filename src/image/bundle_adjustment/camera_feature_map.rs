extern crate nalgebra as na;

use na::DVector;
use std::collections::HashMap;
use crate::image::{
    Image,
    features::Feature,
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
    pub feature_list: Vec<(Point<usize>,Point<usize>)>

}

impl CameraFeatureMap {

    pub fn new<T: Feature>(matches: & Vec<Vec<((usize,T),(usize,T))>>) -> CameraFeatureMap {
        let number_of_unique_features = matches.iter().fold(0,|acc,x| acc + x.len());
        CameraFeatureMap{
            camera_map: HashMap::new(),
            feature_list: Vec::<(Point<usize>,Point<usize>)>::with_capacity(number_of_unique_features)
        }
        
    }

    pub fn add_images_from_params(&mut self, image: &Image, features_per_octave: usize, octave_count: usize) -> () {
        let id = image.id.expect("image has no id");
        self.add_image(id, features_per_octave, octave_count);
    }

    pub fn add_image(&mut self,id: u64, features_per_octave: usize, octave_count: usize) -> () {
        self.camera_map.insert(id,Vec::<(usize,u64)>::with_capacity(features_per_octave*octave_count));

    }

    pub fn add_feature(&mut self, source_cam_id: u64, other_cam_id: u64, 
        x_source: usize, y_source: usize, octave_index_source: usize, 
        x_other: usize, y_other: usize, octave_index_other: usize,  
        pyramid_scale: Float) -> () {

        let (x_source,y_source) = reconstruct_original_coordiantes_for_float(x_source as Float, y_source as Float, pyramid_scale, octave_index_source as i32);
        let (x_other,y_other) = reconstruct_original_coordiantes_for_float(x_other as Float, y_other as Float, pyramid_scale, octave_index_other as i32);
        let point_source = Point::<usize>::new(x_source.trunc() as usize,y_source.trunc() as usize);
        let point_other = Point::<usize>::new(x_other.trunc() as usize,y_other.trunc() as usize);

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

    pub fn add_matches<T: Feature>(&mut self, image_pairs: &Vec<(&Image, &Image)>, matches: & Vec<Vec<((usize,T),(usize,T))>>, pyramid_scale: Float) -> () {
        assert_eq!(image_pairs.len(), matches.len());
        for i in 0..image_pairs.len(){
            let (image_a,image_b) = image_pairs[i];
            let matches_for_pair = &matches[i];

            for ((_,match_a),(_,match_b)) in matches_for_pair {
                let id_a = image_a.id.expect("image a has no id");
                let id_b = image_b.id.expect("image b has no id");


                self.add_feature(id_a, id_b, 
                    match_a.get_x_image(), match_a.get_y_image(),match_a.get_closest_sigma_level(),
                    match_b.get_x_image(), match_b.get_y_image(),match_b.get_closest_sigma_level(),
                    pyramid_scale);
            }

        }

    }

    pub fn get_state(&self) -> State {
        let number_of_cameras = self.camera_map.keys().len();
        //TODO: incorporate transitive associations i.e. f1 -> f1_prime -> f1_alpha is the same
        let number_of_unqiue_points = self.feature_list.len();
        let number_of_cam_parameters = 6*number_of_cameras;
        let number_of_point_parameters = 3*number_of_unqiue_points;
        let total_parameters = number_of_cam_parameters+number_of_point_parameters;
        let data = DVector::<Float>::zeros(total_parameters);
        State{data, n_cams: number_of_cameras, n_points: number_of_unqiue_points}
    }

    /**
     * This vector has ordering In the format [f1_cam1,f2_cam1,f3_cam1,f1_cam2,f2_cam2,...] where cam_id(cam_n-1) < cam_id(cam_n) 
     */
    //TODO: this is wrong -> needs cam information
    pub fn get_observed_features(&self) -> DVector<Float> {
        let n_points = self.feature_list.len();
        let mut observed_features = DVector::<Float>::zeros(n_points*4); // 2 features per point * 2 image coordiantes
        let mut sorted_keys = self.camera_map.keys().cloned().collect::<Vec<u64>>();
        sorted_keys.sort_unstable();
        let feature_offsets = sorted_keys.iter().scan(0,|acc,x| {
            *acc = *acc + self.camera_map.get(x).unwrap().len();
            Some(*acc)
        }).collect::<Vec<usize>>();

        for i in 0..sorted_keys.len() {
            let key = sorted_keys[i];
            let offset = match i {
                0 => 0,
                _ => feature_offsets[i-1]
            };
            let feature_indices = self.camera_map.get(&key).expect(&format!("No image with id: {} found in map",key).to_string());
            for j in 0..feature_indices.len() {
                let (idx, other_cam_id) = feature_indices[j];
                let feature_pos = match (key, other_cam_id) {
                    (s,o) if s < o => self.feature_list[idx].0,
                    _ =>  self.feature_list[idx].1
                };

                observed_features[offset + 2*j] = feature_pos.x as Float;
                observed_features[offset + 2*j+1] = feature_pos.y as Float;
            }
        }
        observed_features
    }



}