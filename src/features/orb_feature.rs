use crate::features::{
    Feature,Oriented,
    fast_feature::FastFeature,
    geometry::point::Point,
    harris_corner::harris_response_for_feature, 
    intensity_centroid,
    orientation
};
use crate::image::Image;
use crate::Float;

#[derive(Debug,Clone,Copy)]
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

impl Oriented for OrbFeature {
    fn get_orientation(&self) -> Float {
        self.orientation
    }
}

impl OrbFeature {

    pub fn new(images: &Vec<Image>, octave_idx: i32, radius: usize, threshold_factor: Float, consecutive_pixels: usize, fast_grid_size: (usize,usize), fast_offsets: (usize,usize), harris_k: Float ) -> Vec<OrbFeature> {
        assert!(images.len() == 1);
        
        let scale_base: Float = 2.0;
        let octave_scale = scale_base.powi(octave_idx);
        let scale_grid_size = ((fast_grid_size.0 as Float * octave_scale).trunc() as usize, (fast_grid_size.1 as Float * octave_scale).trunc() as usize);

        

        let image = &images[0];
        let fast_features = FastFeature::compute_valid_features(image, radius, threshold_factor, consecutive_pixels, scale_grid_size, fast_offsets);
        // Gradient orientation ala SIFT seems to perform better than intensity centroid
        let orientations = fast_features.iter().map(|x| orientation(images, &x.0)).collect::<Vec<Float>>();
        
        let mut indexed_harris_corner_responses = fast_features.iter().map(|x| harris_response_for_feature(images,&x.0,harris_k)).enumerate().collect::<Vec<(usize,Float)>>();
        indexed_harris_corner_responses.sort_unstable_by(|a,b| b.1.partial_cmp(&a.1).unwrap());


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

