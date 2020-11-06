use crate::features::{Feature,fast_feature::FastFeature,intensity_centroid,geometry::point::Point,harris_corner::harris_response_for_feature};
use crate::image::Image;
use crate::Float;

pub struct OrbFeature {
    pub location: Point<usize>,
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

    pub fn new(images: &Vec<Image>, radius: usize, threshold_factor: Float, consecutive_pixels: usize, fast_grid_size: (usize,usize), harris_k: Float ) -> Vec<OrbFeature> {
        assert!(images.len() == 1);

        let image = &images[0];
        let fast_features = FastFeature::compute_valid_features(image, radius, threshold_factor, consecutive_pixels, fast_grid_size);
        let mut indexed_harris_corner_responses = fast_features.iter().map(|x| harris_response_for_feature(images,&x.0,harris_k)).enumerate().collect::<Vec<(usize,Float)>>();
        indexed_harris_corner_responses.sort_unstable_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let orientations = fast_features.iter().map(|x| intensity_centroid::orientation(image, &x.0.get_full_circle().geometry.get_points())).collect::<Vec<Float>>();

        let feature_size = fast_features.len();
        let mut orb_features = Vec::<OrbFeature>::with_capacity(feature_size);

        for i in 0..feature_size {
            let idx = indexed_harris_corner_responses[i].0;
            let orb_feature = OrbFeature {
                location: fast_features[idx].0.location,
                orientation: orientations[idx]
            };
            orb_features.push(orb_feature);
        }
        
        orb_features
    }
}

