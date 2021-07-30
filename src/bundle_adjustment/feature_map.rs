extern crate nalgebra as na;

use na::DMatrix;
use std::collections::HashMap;
use crate::image::Image;
use crate::features::Feature;
use crate::Float;

pub struct FeatureMap {
    pub map: HashMap<u64,(usize,usize,DMatrix<u8>)>

}

impl FeatureMap {

    pub fn new() -> FeatureMap {
        FeatureMap{
            map: HashMap::new()
        }
        
    }

    pub fn add_images_from_params(&mut self, image: &Image, octave_count: usize, pyramid_scale: Float) -> () {

        let (mut width, mut height) = (image.buffer.ncols(), image.buffer.nrows());
        let id = image.id.expect("image has no id");
        for i in 0..octave_count {
            self.add_image(id, width, height, i);
            let (new_width, new_height) = Image::downsampled_dimensions(width, height, pyramid_scale);
            width = new_width;
            height = new_height;
        }


    }

    pub fn add_image(&mut self,id: u64, width: usize, height: usize, octave_count: usize) -> () {
        let pixel_count = width*height;
        self.map.insert(id,(width,height,DMatrix::<u8>::zeros(pixel_count,octave_count)));

    }

    pub fn add_feature(&mut self, id: u64, x: usize, y: usize, octave_index: usize) -> () {
        let map_element = self.map.get_mut(&id).expect(&format!("No image with id: {} found in map",id).to_string());
        let (_,height, map) = map_element;
        let h = *height;
        let pixel_id = y*h+x;
        map[(pixel_id,octave_index)] = 1;
        
    }

    pub fn remove_feature(&mut self, id: u64, x: usize, y: usize, octave_index: usize) -> () {
        let map_element = self.map.get_mut(&id);
        if map_element.is_some() {
            let (_,height, map) = map_element.unwrap();
            let h = *height;
            let pixel_id = y*h+x;
            map[(pixel_id,octave_index)] = 0;
        }
    }

    pub fn add_matches<T: Feature>(&mut self, image_pairs: &Vec<(&Image, &Image)>, matches: & Vec<Vec<((usize,T),(usize,T))>>) -> () {
        assert_eq!(image_pairs.len(), matches.len());
        for i in 0..image_pairs.len(){
            let (image_a,image_b) = image_pairs[i];
            let matches_for_pair = &matches[i];

            for ((_,match_a),(_,match_b)) in matches_for_pair {
                let id_a = image_a.id.expect("image a has no id");
                let id_b = image_b.id.expect("image b has no id");

                self.add_feature(id_a, match_a.get_x_image(), match_a.get_y_image(),match_a.get_closest_sigma_level());
                self.add_feature(id_b, match_b.get_x_image(), match_b.get_y_image(),match_b.get_closest_sigma_level());
            }

        }

    }



}