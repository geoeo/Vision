extern crate nalgebra as na;

use na::DVector;
use std::collections::HashMap;
use crate::image::Image;
use crate::image::features::Feature;
use crate::{Float, reconstruct_original_coordiantes_for_float};
use crate::image::features::geometry::point::Point;

pub struct CameraFeatureMap {
    pub camera_map: HashMap<u64, Vec<(usize,u64)>>,
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

    pub fn get_state(&self) -> (DVector<Float>, usize, usize) {
        let number_of_cameras = self.camera_map.keys().len();
        let number_of_unqiue_points = self.feature_list.len();
        let number_of_cam_parameters = 6*number_of_cameras;
        let number_of_point_parameters = 3*number_of_unqiue_points;
        let total_parameters = number_of_cam_parameters*number_of_point_parameters;
        let state = DVector::<Float>::zeros(total_parameters);
        (state, number_of_cameras,number_of_unqiue_points)
    }



}