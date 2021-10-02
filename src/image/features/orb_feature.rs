use serde::{Serialize, Deserialize};
use crate::image::features::{
    Feature,Oriented,
    fast_feature::FastFeature,
    geometry::point::Point,
    harris_corner::harris_response_for_feature, 
    orientation,
};
use crate::image::pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters;
use crate::image::Image;
use crate::Float;

#[derive(Debug,Clone,Copy,Serialize, Deserialize)]
pub struct OrbFeature {
    pub location: Point<usize>,
    pub orientation: Float,
    pub sigma_level: usize
}

impl Feature for OrbFeature {
    fn get_x_image(&self) -> usize {
        self.location.x
    }

    fn get_y_image(&self) -> usize {
        self.location.y
    }

    fn get_closest_sigma_level(&self) -> usize {
        self.sigma_level
    }


}

impl Oriented for OrbFeature {
    fn get_orientation(&self) -> Float {
        self.orientation
    }
}

impl OrbFeature {

    pub fn new(images: &Vec<Image>, octave_idx: i32, runtime_parameters: &OrbRuntimeParameters) -> Vec<OrbFeature> {
        assert!(images.len() == 1);

        
        let image = &images[0];
        let fast_features = FastFeature::compute_valid_features(image,octave_idx, runtime_parameters,);
        // Gradient orientation ala SIFT seems to perform better than intensity centroid => move this to feature
        let orientations = fast_features.iter().map(|x| orientation(images, x)).collect::<Vec<Float>>();
        
        let mut indexed_harris_corner_responses = fast_features.iter().map(|x| harris_response_for_feature(images,x,runtime_parameters.harris_k, runtime_parameters.harris_window_size)).enumerate().collect::<Vec<(usize,Float)>>();
        indexed_harris_corner_responses.sort_unstable_by(|a,b| b.1.partial_cmp(&a.1).unwrap());


        let feature_size = fast_features.len();
        let mut orb_features = Vec::<OrbFeature>::with_capacity(feature_size);

        for i in 0..feature_size {
            let idx = indexed_harris_corner_responses[i].0;
            let location =  fast_features[idx].location;

            let orb_feature = OrbFeature {
                location,
                orientation: orientations[idx],
                sigma_level: octave_idx as usize
            };
            orb_features.push(orb_feature);
        }
        
        orb_features
    }
}

