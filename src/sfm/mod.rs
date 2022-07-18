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
    pub fn matches(&self) -> &Vec<Vec<Vec<Match<F>>>> { &self.matches } // TODO: These are not the filtered matches which are usually what are used. Unify this

    pub fn compute_path_id_pairs(&self) -> Vec<Vec<(usize, usize)>> {
        let mut path_id_paris = Vec::<Vec::<(usize,usize)>>::with_capacity(self.paths.len());
        for sub_path in &self.paths {
            path_id_paris.push(
                sub_path.iter().enumerate().map(|(i,&id)| 
                    match i {
                        0 => (self.root,id),
                        idx => (sub_path[idx-1],id)
                    }
                ).collect()
            )
        }

        path_id_paris
    }

    pub fn compute_unqiue_ids_cameras_root_first(&self) -> (Vec<usize>, Vec<&C>) {
        let number_of_keys = self.camera_map.keys().len();
        let mut keys_sorted = Vec::<usize>::with_capacity(number_of_keys);
        // root has to first by design
        keys_sorted.push(self.root());
        keys_sorted.extend(self.paths.clone().into_iter().flatten().collect::<Vec<usize>>());
        keys_sorted.dedup();
        let cameras_sorted = keys_sorted.iter().map(|id| self.camera_map.get(id).expect("compute_unqiue_ids_cameras_sorted: trying to get invalid camera")).collect::<Vec<&C>>();
        (keys_sorted,cameras_sorted)
    }

}