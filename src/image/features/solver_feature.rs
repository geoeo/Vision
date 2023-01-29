extern crate nalgebra as na;

use na::Vector2;
use crate::image::features::{ImageFeature, orb_feature::OrbFeature};
use crate::image::features::geometry::point::Point;
use crate::Float;

pub trait SolverFeature {
    fn empty() -> Self where Self: Sized;
    fn update(&mut self, new_value: &Vector2<Float>) -> ();
}

impl SolverFeature for ImageFeature { 
    fn empty() -> ImageFeature { ImageFeature {location: Point {x: 0.0, y: 0.0}} }
    fn update(&mut self, new_value: &Vector2<Float>) {
        self.location.x = new_value[0];
        self.location.y = new_value[1]; 
    }
}

impl SolverFeature for OrbFeature { 
    fn empty() -> OrbFeature { OrbFeature {location: Point {x: 0, y: 0}, orientation: 0.0, sigma_level: 0 } }
    fn update(&mut self, new_value: &Vector2<Float>) {
        self.location.x = new_value[0] as usize;
        self.location.y = new_value[1] as usize;
    }
}