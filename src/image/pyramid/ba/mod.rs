pub mod ba_octave;

use crate::image::features::Feature;

pub struct BAOctave<F: Feature> {
    pub features: Vec<F>,
}