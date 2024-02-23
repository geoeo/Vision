
use std::collections::HashMap;

use crate::image::features::Feature;

#[derive(Debug,Clone)]
pub struct BAOctave {
    pub level: usize, // base is 0
    pub grid: HashMap<(usize,usize),Vec<usize>>
}

impl BAOctave {

    pub fn new<F: Feature>(level: usize, features: &Vec<F>, image_width: usize, image_height: usize) -> BAOctave {
        let number_of_features = features.len();
        let (grid_size_width, grid_size_height) = BAOctave::get_grid_size(level, image_width, image_height);
        let grid_capacity = (image_width/grid_size_width) * (image_height/grid_size_height);
        let mut grid =  HashMap::<(usize,usize), Vec::<usize>>::with_capacity(grid_capacity);

        for i in (0..image_width).step_by(grid_size_width) {
            for j in (0..image_height).step_by(grid_size_height) {
                let key = (i/grid_size_width,j/grid_size_height);
                grid.insert(key, Vec::<usize>::with_capacity(number_of_features));
            }
        }

        for (idx, feat) in features.iter().enumerate() {
            let id = Self::get_grid_id(feat,level,image_width,image_height);
            grid.get_mut(&id).unwrap().push(idx);
        }

        BAOctave {level, grid}
    }

    fn get_grid_size(level: usize, image_width: usize, image_height: usize) -> (usize,usize) {
        let step = 2*(level+1);
        (image_width/step, image_height/step)
    }

    fn get_grid_id<F: Feature>(feature: &F, level: usize, image_width: usize, image_height: usize) -> (usize,usize){
        let (grid_size_width, grid_size_height) = BAOctave::get_grid_size(level,image_width,image_height);
        let feature_pos_x = feature.get_x_image();
        let feature_pos_y = feature.get_y_image();
        let x_id = feature_pos_x/grid_size_width;
        let y_id = feature_pos_y/grid_size_height;
        (x_id,y_id)
    }

    pub fn calc_score(&self) -> usize {
        self.grid.values().map(|v| v.len()).sum()
    }
}