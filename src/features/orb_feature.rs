use crate::features::{
    Feature,Oriented,
    fast_feature::FastFeature,
    geometry::point::Point,
    harris_corner::harris_response_for_feature, 
    intensity_centroid,
    orientation,
};
use crate::pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters;
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

    pub fn new(images: &Vec<Image>, octave_idx: i32, runtime_parameters: &OrbRuntimeParameters) -> Vec<OrbFeature> {
        assert!(images.len() == 1);

        let orig_offset = runtime_parameters.fast_offsets;
        let offset_scale = runtime_parameters.fast_offset_scale_base.powi(octave_idx) as Float;
        let x_offset_scaled = (orig_offset.0 as Float / offset_scale).trunc() as usize;
        let y_offset_scaled = (orig_offset.1 as Float / offset_scale).trunc() as usize;
        
        let octave_scale = runtime_parameters.fast_grid_size_scale_base.powi(octave_idx);
        let scale_grid_size = ((runtime_parameters.fast_grid_size.0 as Float * octave_scale).trunc() as usize, (runtime_parameters.fast_grid_size.1 as Float * octave_scale).trunc() as usize);

        

        let image = &images[0];
        let fast_features = FastFeature::compute_valid_features(image, runtime_parameters.fast_circle_radius, runtime_parameters.fast_threshold_factor, runtime_parameters.fast_consecutive_pixels, scale_grid_size, (x_offset_scaled,y_offset_scaled));
        // Gradient orientation ala SIFT seems to perform better than intensity centroid => move this to feature
        let orientations = fast_features.iter().map(|x| orientation(images, &x.0)).collect::<Vec<Float>>();
        //TODO: seems buggy
        //let orientations = fast_features.iter().map(|x| intensity_centroid::orientation(&images[0], &x.0.get_all_points_in_radius())).collect::<Vec<Float>>();
        

        //TODO: this might crash with bad parameter setup
        let mut indexed_harris_corner_responses = fast_features.iter().map(|x| harris_response_for_feature(images,&x.0,runtime_parameters.harris_k, runtime_parameters.harris_window_size)).enumerate().collect::<Vec<(usize,Float)>>();
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

