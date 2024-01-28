use crate::image::features::Feature;
use crate::image::pyramid::ba::ba_octave::BAOctave;

#[derive(Debug,Clone)]
pub struct BAPyramid<F: Feature> {
    features: Vec<F>,
    levels: usize,
    image_width: usize,
    image_height: usize
}

impl<F: Feature> BAPyramid<F> {

    pub fn new(features: &Vec<F>, levels: usize, image_width: usize, image_height: usize) -> BAPyramid<F> {
        BAPyramid {features: features.clone(), levels, image_width,image_height}
    }

    pub fn add_features(&mut self, features: &Vec<F>) -> () {
        self.features.extend(features.clone());
    }

    pub fn calculate_score(&self) -> usize {
        (0..self.levels).map(|level| BAOctave::new(level, &self.features, self.image_width, self.image_height)).map(|oct| oct.calc_score()).sum()
    }

}