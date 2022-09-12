
use crate::image::features::ImageFeature;
use crate::image::features::geometry::point::Point;

pub trait SolverFeature {
    fn empty() -> Self where Self: Sized;
    fn add(&mut self, inc: &Self) -> ();
}

impl SolverFeature for ImageFeature { 
    fn empty() -> ImageFeature { ImageFeature {location: Point {x: 0.0, y: 0.0} } }
    fn add(&mut self, inc: &Self) {
        self.location.x += inc.location.x;
        self.location.y += inc.location.y;
    }
}