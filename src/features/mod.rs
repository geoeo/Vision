pub mod geometry;
pub mod sift_feature;
pub mod fast_feature;
pub mod octave_feature;

pub trait Feature {
    fn get_x_image(&self) -> usize;
    fn get_y_image(&self) -> usize;
    fn get_closest_sigma_level(&self) -> usize;
}