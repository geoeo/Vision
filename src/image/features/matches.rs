use nalgebra as na;

use na::Matrix3;
use serde::{Serialize, Deserialize};
use crate::Float;
use crate::image::features::Feature;
use std::marker::{Send,Sync};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Match<T : Feature + Send + Sync> {
    feature_one: T,
    feature_two: T
}

impl<T: Feature + Send + Sync> Match<T> {
    pub fn new(feature_one: T, feature_two: T) -> Match<T> {
        assert_eq!(feature_one.get_landmark_id(), feature_two.get_landmark_id());
        Match {feature_one, feature_two}
    }
    pub fn apply_normalisation(&self, norm_one: &Matrix3<Float>, norm_two: &Matrix3<Float>, depth: Float) -> Self {
        let feature_one = self.feature_one.apply_normalisation(norm_one, depth);
        let feature_two = self.feature_two.apply_normalisation(norm_two, depth);
        Match {feature_one, feature_two}
    }
    pub fn get_landmark_id(&self) -> Option<usize> {self.feature_one.get_landmark_id()}
    pub fn set_landmark_id(&mut self, new_landmark_id: Option<usize>) {
        self.feature_one = self.feature_one.copy_with_landmark_id(new_landmark_id);
        self.feature_two = self.feature_two.copy_with_landmark_id(new_landmark_id);
    }
    pub fn get_feature_one(&self) -> &T {&self.feature_one}
    pub fn get_feature_two(&self) -> &T {&self.feature_two}
    pub fn get_feature_one_mut(&mut self) -> &mut T {&mut self.feature_one}
    pub fn get_feature_two_mut(&mut self) -> &mut T {&mut self.feature_two}
}


impl<T: Feature + PartialEq + Send + Sync> PartialEq for Match<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.feature_one == other.feature_one) && (self.feature_two == other.feature_two)
    }
}

//TODO
pub fn subsample_matches<T: Feature + Clone + Send + Sync>(matches: Vec<Match<T>>, _: usize, _: usize) -> Vec<Match<T>> {
    matches
}