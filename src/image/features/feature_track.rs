use crate::image::features::{Match,Feature};

pub struct FeatureTrack<T: Feature> {
    track: Vec<Match<T>>
}

impl<T: Feature + Clone> FeatureTrack<T> {
    pub fn new(capacity: usize, m: &Match<T>) -> FeatureTrack<T> {
        let mut track = Vec::<Match<T>>::with_capacity(capacity);
        track.push(m.clone());
        FeatureTrack{track}
    }


    pub fn add(&mut self, m: &Match<T>) -> () {
        self.track.push(m.clone());
    }

    /**
     * Returns a Tuple of the current image coordiantes as (x,y) format
     */
    pub fn get_current_id(&self) -> (usize, usize) {
        let feature_two = &self.track.last().expect("FeatureTrack: Called get_current_id on empty track").feature_two;
        (feature_two.get_x_image(),feature_two.get_y_image())
    }
}