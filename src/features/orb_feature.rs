use crate::features::{Feature,fast_feature::FastFeature,intensity_centroid,geometry::point::Point};
use crate::image::Image;
use crate::Float;

pub struct OrbFeature {
    pub location: Point,
    pub orientation: Float
}

impl Feature for OrbFeature {
    fn get_x_image(&self) -> usize {
        self.location.x
    }

    fn get_y_image(&self) -> usize {
        self.location.y
    }

    fn get_closest_sigma_level(&self) -> usize {
        0
    }
}

impl OrbFeature {

    //TODO: gridsize for Fast non max suppression??
    pub fn new(image: &Image, radius: usize,  threshold_factor: Float, consecutive_pixels: usize, fast_grid_size: (usize,usize) ) -> Vec<OrbFeature> {

        let fast_features = FastFeature::compute_valid_features(image, radius, threshold_factor, consecutive_pixels, fast_grid_size);
        let orientations = fast_features.iter().map(|x| intensity_centroid::orientation(image, &x.0.get_full_circle().geometry.get_points())).collect::<Vec<Float>>();

        let mut orb_features = Vec::<OrbFeature>::with_capacity(fast_features.len());

        for i in 0..fast_features.len() {
            let orb_feature = OrbFeature {
                location: fast_features[i].0.location,
                orientation: orientations[i]
            };
            orb_features.push(orb_feature);
        }
        
        orb_features
    }
}

