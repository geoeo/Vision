use crate::image::pyramid::orb::OrbOctave;

#[derive(Debug,Clone)]
pub struct OrbPyramid {
    pub octaves: Vec<OrbOctave>
}

impl OrbPyramid {
    pub fn empty(number_of_octave: usize) -> OrbPyramid {
        OrbPyramid {octaves: Vec::<OrbOctave>::with_capacity(number_of_octave)}
    }
}