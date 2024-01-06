use crate::image::features::{matches::Match,Feature};
use std::marker::{Send,Sync};

type PathIdx = usize;
type ImageIdx = usize;

pub struct FeatureTrack<T: Feature + Send + Sync> {
    track: Vec<(PathIdx, ImageIdx, Match<T>)>,
    landmark_id: usize
}

impl<T: Feature + Clone + PartialEq + Send + Sync> FeatureTrack<T> {
    pub fn new(capacity: usize, path_idx: PathIdx, img_idx: ImageIdx, landmark_id: usize, m: &Match<T>) -> FeatureTrack<T> {
        let mut track = Vec::<(PathIdx, ImageIdx, Match<T>)>::with_capacity(capacity);
        let new_m = Match::new(m.get_feature_one().copy_with_landmark_id(Some(landmark_id)),m.get_feature_two().copy_with_landmark_id(Some(landmark_id)));
        track.push((path_idx, img_idx, new_m));
        FeatureTrack{track, landmark_id}
    }


    pub fn add(&mut self, path_idx: PathIdx, image_idx: ImageIdx, m: &Match<T>) -> () {
        let new_m = Match::new(m.get_feature_one().copy_with_landmark_id( Some(self.landmark_id)), m.get_feature_two().copy_with_landmark_id( Some(self.landmark_id)));
        self.track.push((path_idx, image_idx, new_m));
    }

    /**
     * Returns current feature
     */
    pub fn get_current_feature_dest(&self) -> T {
        self.track.last().expect("FeatureTrack: Called get_current_id on empty track").2.get_feature_two().clone()
    }

    pub fn get_first_feature_start(&self) -> T {
        self.track.first().expect("FeatureTrack: Called get_first_feature on empty track").2.get_feature_one().clone()
    }

    /**
     * Returns (path_idx, image_idx)
     */
    pub fn get_path_img_id(&self) -> (usize, usize) {
        let t = self.track.last().expect("FeatureTrack: Called get_current_id on empty track");
        (t.0, t.1)
    }

    pub fn get_track_length(&self) -> usize {
        let max_track_len = self.track.capacity();
        let start_offset = self.track.first().expect("FeatureTrack: Called get_track_length on empty track").1;
        
        match self.track.len() {
            l if l > 2 && l < max_track_len => l + start_offset,
            l => l
        }
    }

    pub fn get_track(&self) -> &Vec<(PathIdx, ImageIdx, Match<T>)> {
        &self.track
    }

    pub fn get_track_id(&self) -> usize {
        self.landmark_id
    }

}

