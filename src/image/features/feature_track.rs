use crate::image::features::{Match,Feature};
type PathIdx = usize;
type ImageIdx = usize;

pub struct FeatureTrack<T: Feature> {
    track: Vec<(PathIdx, ImageIdx, Match<T>)>
}

impl<T: Feature + Clone> FeatureTrack<T> {
    pub fn new(capacity: usize,path_idx: usize , m: &Match<T>) -> FeatureTrack<T> {
        let mut track = Vec::<(PathIdx, ImageIdx, Match<T>)>::with_capacity(capacity);
        track.push((path_idx ,0, m.clone()));
        FeatureTrack{track}
    }


    pub fn add(&mut self, path_idx: PathIdx, image_idx: ImageIdx, m: &Match<T>) -> () {
        self.track.push((path_idx, image_idx, m.clone()));
    }

    /**
     * Returns a Tuple of the current image coordiantes as (x,y) format
     */
    pub fn get_feature_current_id(&self) -> (usize, usize) {
        let feature_two = &self.track.last().expect("FeatureTrack: Called get_current_id on empty track").2.feature_two;
        (feature_two.get_x_image(),feature_two.get_y_image())
    }

        /**
     * Returns (path_idx, image_idx)
     */
    pub fn get_path_img_id(&self) -> (usize, usize) {
        let t = &self.track.last().expect("FeatureTrack: Called get_current_id on empty track");
        (t.0, t.1)
    }

    pub fn get_track_length(&self) -> usize {
        self.track.len()
    }

    pub fn get_track(&self) -> &Vec<(PathIdx, ImageIdx, Match<T>)> {
        &self.track
    }
}

