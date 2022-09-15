extern crate nalgebra as na;

use na::Vector2;
use crate::image::features::ImageFeature;
use crate::image::features::geometry::point::Point;
use crate::Float;

pub trait SolverFeature {
    fn empty() -> Self where Self: Sized;
    fn update(&mut self, new_value: &Vector2<Float>) -> ();
}

impl SolverFeature for ImageFeature { 
    fn empty() -> ImageFeature { ImageFeature {location: Point {x: 0.0, y: 0.0} } }
    fn update(&mut self, new_value: &Vector2<Float>) {
        self.location.x = new_value[0];
        self.location.y = new_value[1];
    }
}