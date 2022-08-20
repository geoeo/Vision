use crate::image::features::{Match,Feature};

pub struct FeatureTrack<T: Feature> {
    track: Vec<Match<T>>
}

impl<T: Feature + Clone> FeatureTrack<T> {
    pub fn new(capacity: usize) -> FeatureTrack<T> {
        FeatureTrack{track: Vec::<Match<T>>::with_capacity(capacity)}
    }

    pub fn add(&mut self, m: &Match<T>) -> () {
        self.track.push(m.clone());
    }

    pub fn get_current_id(&self) -> (usize, usize) {
        let feature_two = &self.track.last().expect("FeatureTrack: Called get_current_id on empty track").feature_two;
        (feature_two.get_x_image(),feature_two.get_y_image())
    }
}