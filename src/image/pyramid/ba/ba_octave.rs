
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
        let grid_size = BAOctave::get_grid_size(level);
        let grid_capacity = (image_width/grid_size) * (image_height/grid_size);
        let mut grid =  HashMap::with_capacity(grid_capacity);

        for i in 0..grid_size {
            for j in 0..grid_size {
                grid.insert((i,j), Vec::<usize>::with_capacity(number_of_features));
            }
        }

        for (idx, feat) in features.iter().enumerate() {
            let id = Self::get_grid_id(feat,level);
            grid.get_mut(&id).unwrap().push(idx);
        }

        BAOctave {level, grid}
    }

    fn get_grid_size(level: usize) -> usize {
        2*(level+1)
    }

    fn get_grid_id<F: Feature>(feature: &F, level: usize) -> (usize,usize){
        let grid_size = BAOctave::get_grid_size(level);
        let feature_pos_x = feature.get_x_image();
        let feature_pos_y = feature.get_y_image();
        let x_id = feature_pos_x/grid_size;
        let y_id = feature_pos_y/grid_size;
        (x_id,y_id)
    }

    pub fn calc_score(&self) -> usize {
        self.grid.values().map(|v| v.len()).sum()
    }
}