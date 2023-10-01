use nalgebra as na;

use na::Matrix3;
use std::hash::{Hash, Hasher};
use crate::Float;
use crate::image::features::Feature;
use crate::image::features::geometry::point::Point;

#[derive(Clone,Eq)]
pub struct ImageFeature {
    pub location: Point<Float>,
    pub landmark_id: Option<usize>
}

impl PartialEq for ImageFeature {
    fn eq(&self, other: &Self) -> bool {
        self.get_location() == other.get_location()
    }
}

impl Hash for ImageFeature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x = self.get_x_image();
        let y = self.get_y_image();
        (x,y).hash(state);
    }
}

impl ImageFeature {
    pub fn new(x: Float, y: Float, landmark_id: Option<usize>) -> ImageFeature { ImageFeature{location: Point::new(x, y), landmark_id} }
}

impl Feature for ImageFeature {
    fn new(x: Float, y: Float, landmark_id: Option<usize>) -> ImageFeature { ImageFeature{location: Point::new(x, y), landmark_id} }
    fn get_location(&self) -> Point<Float> { self.location }
    fn get_x_image_float(&self) -> Float { self.location.x }
    fn get_y_image_float(&self) -> Float { self.location.y }
    fn get_x_image(&self) -> usize { self.location.x.trunc() as usize}
    fn get_y_image(&self) -> usize { self.location.y.trunc() as usize}
    fn get_closest_sigma_level(&self) -> usize {0}
    fn apply_normalisation(&self, norm: &Matrix3<Float>, depth: Float) -> Self {
        let v = norm*self.get_as_homogeneous(depth);
        ImageFeature::new(v[0], v[1], self.get_landmark_id())
    }

    fn get_landmark_id(&self) -> Option<usize> {
        self.landmark_id
    }
    fn copy_with_landmark_id(&self, landmark_id: Option<usize>) -> Self {
        ImageFeature::new(self.get_x_image_float(), self.get_y_image_float(), landmark_id)
    }
}