use crate::features::{Feature,fast_feature::FastFeature};
use crate::image::Image;
use crate::Float;

pub struct OrbFeature {
    pub fast_feature: FastFeature,
    pub orientation: Float,
    pub pyramid_level: usize
}

impl Feature for OrbFeature {
    fn get_x_image(&self) -> usize {
        self.fast_feature.x_center
    }

    fn get_y_image(&self) -> usize {
        self.fast_feature.y_center
    }

    fn get_closest_sigma_level(&self) -> usize {
        0
    }
}

impl OrbFeature {

    //TODO: gridsize for Fast non max suppression??
    pub fn new(image: &Image, radius: usize,  threshold_factor: Float, n: usize, fast_grid_size: (usize,usize) ) -> OrbFeature {
        panic!("Not yet implemented")

    }
}

