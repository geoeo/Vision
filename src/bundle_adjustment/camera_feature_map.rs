extern crate nalgebra as na;

use na::DVector;
use std::collections::HashMap;
use crate::image::Image;
use crate::features::Feature;
use crate::{Float, reconstruct_original_coordiantes_for_float};
use crate::features::geometry::point::Point;

pub struct CameraFeatureMap {
    pub map: HashMap<u64,Vec<Point<usize>>> //Maybe keep dimensiosn fo feature state init

}

impl CameraFeatureMap {

    pub fn new() -> CameraFeatureMap {
        CameraFeatureMap{
            map: HashMap::new()
        }
        
    }

    pub fn add_images_from_params(&mut self, image: &Image, features_per_octave: usize, octave_count: usize) -> () {
        let id = image.id.expect("image has no id");
        self.add_image(id, features_per_octave, octave_count);
    }

    pub fn add_image(&mut self,id: u64, features_per_octave: usize, octave_count: usize) -> () {
        self.map.insert(id,Vec::<Point<usize>>::with_capacity(features_per_octave*octave_count));

    }

    pub fn add_feature(&mut self, id: u64, x: usize, y: usize, octave_index: usize, pyramid_scale: Float) -> () {
        let map = self.map.get_mut(&id).expect(&format!("No image with id: {} found in map",id).to_string());
        let (x,y) = reconstruct_original_coordiantes_for_float(x as Float, y as Float, pyramid_scale, octave_index as i32);


        map.push(Point::<usize>::new(x.trunc() as usize,y.trunc() as usize));
        
    }


    pub fn add_matches<T: Feature>(&mut self, image_pairs: &Vec<(&Image, &Image)>, matches: & Vec<Vec<((usize,T),(usize,T))>>, pyramid_scale: Float) -> () {
        assert_eq!(image_pairs.len(), matches.len());
        for i in 0..image_pairs.len(){
            let (image_a,image_b) = image_pairs[i];
            let matches_for_pair = &matches[i];

            for ((_,match_a),(_,match_b)) in matches_for_pair {
                let id_a = image_a.id.expect("image a has no id");
                let id_b = image_b.id.expect("image b has no id");

                self.add_feature(id_a, match_a.get_x_image(), match_a.get_y_image(),match_a.get_closest_sigma_level(), pyramid_scale);
                self.add_feature(id_b, match_b.get_x_image(), match_b.get_y_image(),match_b.get_closest_sigma_level(), pyramid_scale);
            }

        }

    }

    pub fn get_state(&self) -> DVector<Float> {
        let camera_ids = self.map.keys();
        let number_of_cameras = camera_ids.len();
        let number_of_features_per_cam = self.map.values().map(|list| list.len()).collect::<Vec<usize>>();
        let number_of_features :usize = number_of_features_per_cam.iter().sum();
        let number_of_cam_parameters = 3*number_of_cameras;
        let number_of_point_parameters = 3*number_of_features;
        let total_parameters = number_of_cam_parameters*number_of_point_parameters;
        let state = DVector::<Float>::zeros(total_parameters);
        state
    }



}