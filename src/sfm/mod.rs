use std::collections::HashMap;
use crate::sensors::camera::Camera;
use crate::image::features::{Feature, Match};

pub mod bundle_adjustment;
pub mod landmark;

/**
 * We assume that the indices between paths and matches are consistent
 */
pub struct SFMConfig<C: Camera, F: Feature> {
    root: usize,
    paths: Vec<Vec<usize>>,
    camera_map: HashMap<usize, C>,
    matches: Vec<Vec<Vec<Match<F>>>>
}

impl<C: Camera, F: Feature> SFMConfig<C,F> {

    pub fn new(root: usize, paths: Vec<Vec<usize>>, camera_map: HashMap<usize, C>, matches: Vec<Vec<Vec<Match<F>>>>) -> SFMConfig<C,F> {
        SFMConfig{root, paths, camera_map, matches}
    }

    pub fn root(&self) -> usize { self.root }
    pub fn paths(&self) -> &Vec<Vec<usize>> { &self.paths }
    pub fn camera_map(&self) -> &HashMap<usize, C> { &self.camera_map }
    pub fn matches(&self) -> &Vec<Vec<Vec<Match<F>>>> { &self.matches }

}