extern crate nalgebra as na;

use na::DMatrix;
use std::collections::HashMap;
use crate::image::Image;
use crate::features::Feature;
use crate::{Float, reconstruct_original_coordiantes_for_float};

pub struct FeatureMap {
    pub map: HashMap<u64,DMatrix<u8>>

}

impl FeatureMap {

    pub fn new() -> FeatureMap {
        FeatureMap{
            map: HashMap::new()
        }
        
    }

    pub fn add_images_from_params(&mut self, image: &Image) -> () {

        let (width, height) = (image.buffer.ncols(), image.buffer.nrows());
        let id = image.id.expect("image has no id");
        self.add_image(id, width, height);
    }

    pub fn add_image(&mut self,id: u64, width: usize, height: usize) -> () {
        self.map.insert(id,DMatrix::<u8>::zeros(height,width));

    }

    pub fn add_feature(&mut self, id: u64, x: usize, y: usize, octave_index: usize, pyramid_scale: Float) -> () {
        let map = self.map.get_mut(&id).expect(&format!("No image with id: {} found in map",id).to_string());
        let (x,y) = reconstruct_original_coordiantes_for_float(x as Float, y as Float, pyramid_scale, octave_index as i32);


        map[(y.trunc() as usize,x.trunc() as usize)] = 1;
        
    }

    pub fn remove_feature(&mut self, id: u64, x: usize, y: usize, octave_index: usize, pyramid_scale: Float) -> () {
        let map = self.map.get_mut(&id).expect(&format!("No image with id: {} found in map",id).to_string());
        let (x,y) = reconstruct_original_coordiantes_for_float(x as Float, y as Float, pyramid_scale, octave_index as i32);
        let x_usize = x.trunc() as usize;
        let y_usize = y.trunc() as usize;

        map[(y_usize,x_usize)] = 0;
        

        
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



}