use crate::image::features::Feature;
use crate::image::pyramid::ba::ba_octave::BAOctave;

#[derive(Debug,Clone)]
pub struct BAPyramid<F: Feature> {
    pub features: Vec<F>,
    pub octaves: Vec<BAOctave>
}

impl<F: Feature> BAPyramid<F> {

    pub fn new(features: &Vec<F>, levels: usize, image_width: usize, image_height: usize) -> BAPyramid<F> {
        let octaves = (0..levels).map(|level| BAOctave::new(level, features, image_width, image_height)).collect();
        BAPyramid {features: features.clone(),octaves}
    }

    pub fn calculate_score(&self) -> usize {
        self.octaves.iter().map(|oct| oct.calc_score()).sum()
    }

}