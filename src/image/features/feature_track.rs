use crate::image::features::{Match,Feature};
type PathIdx = usize;
type ImageIdx = usize;

pub struct FeatureTrack<T: Feature> {
    track: Vec<(PathIdx, ImageIdx, Match<T>)>
}

impl<T: Feature + Clone + PartialEq> FeatureTrack<T> {
    pub fn new(capacity: usize,path_idx: usize , m: &Match<T>) -> FeatureTrack<T> {
        let mut track = Vec::<(PathIdx, ImageIdx, Match<T>)>::with_capacity(capacity);
        track.push((path_idx ,0, m.clone()));
        FeatureTrack{track}
    }


    pub fn add(&mut self, path_idx: PathIdx, image_idx: ImageIdx, m: &Match<T>) -> () {
        self.track.push((path_idx, image_idx, m.clone()));
    }

    /**
     * Returns current feature
     */
    pub fn get_current_feature(&self) -> T {
        self.track.last().expect("FeatureTrack: Called get_current_id on empty track").2.feature_two.clone()
    }

    /**
     * Returns (path_idx, image_idx)
     */
    pub fn get_path_img_id(&self) -> (usize, usize) {
        let t = self.track.last().expect("FeatureTrack: Called get_current_id on empty track");
        (t.0, t.1)
    }

    pub fn get_track_length(&self) -> usize {
        self.track.len() + self.track.first().expect("FeatureTrack: Called get_track_length on empty track").1
    }

    pub fn get_track(&self) -> &Vec<(PathIdx, ImageIdx, Match<T>)> {
        &self.track
    }
}

