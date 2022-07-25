use nalgebra as na;

use na::{Vector2,Vector3,Matrix3, OMatrix, base::dimension::U3, base::dimension::Dynamic};
use serde::{Serialize, Deserialize};
use crate::{Float,float};
use crate::image::Image;
use crate::image::filter::{prewitt_kernel::PrewittKernel,gradient_convolution_at_sample};
use crate::image::features::geometry::point::Point;
use crate::GradientDirection;

pub mod geometry;
pub mod sift_feature;
pub mod fast_feature;
pub mod harris_corner;
pub mod hessian_response;
pub mod orb_feature;
pub mod intensity_centroid;


pub trait Feature {
    fn get_x_image_float(&self) -> Float;
    fn get_y_image_float(&self) -> Float;
    fn get_x_image(&self) -> usize;
    fn get_y_image(&self) -> usize;
    fn get_closest_sigma_level(&self) -> usize;
    fn reconstruct_original_coordiantes_for_float(&self, pyramid_scaling: Float) -> (Float, Float) {
        let factor = pyramid_scaling.powi(self.get_closest_sigma_level() as i32);
        ((self.get_x_image_float() as Float)*factor, (self.get_y_image_float() as Float)*factor)
    }
    fn get_as_3d_point(&self, depth: Float) -> Vector3<Float> {
       Vector3::<Float>::new(self.get_x_image_float(), self.get_y_image_float(), depth)
    }
    fn get_as_2d_point(&self) -> Vector2<Float> {
        Vector2::<Float>::new(self.get_x_image_float(), self.get_y_image_float())
    }

    /**
     * Gets the camera ray for image points which are assumed to lie on the focal plane with depth +- 1
     */
    fn get_camera_ray(&self, inverse_intrinsics: &Matrix3<Float>) -> Vector3<Float> {
        inverse_intrinsics*Vector3::<Float>::new(self.get_x_image_float(), self.get_y_image_float(),1.0)
    }
}

pub trait Oriented {
    fn get_orientation(&self) -> Float;
}

pub fn orientation(source_images: &Vec<Image>, feature: &dyn Feature) -> Float {
    let kernel = PrewittKernel::new();
    let x_grad = gradient_convolution_at_sample(source_images,feature, &kernel, GradientDirection::HORIZINTAL);
    // We negate here because the y-axis of a matrix is inverted from the first quadrant of a cartesian plane
    let y_grad = -gradient_convolution_at_sample(source_images,feature, &kernel, GradientDirection::VERTICAL);
    match y_grad.atan2(x_grad) {
        angle if angle < 0.0 => 2.0*float::consts::PI + angle,
        angle => angle
    }

}

#[derive(Clone)]
pub struct ImageFeature {
    pub location: Point<Float>
}

impl ImageFeature {
    pub fn new(x: Float, y: Float) -> ImageFeature {
        ImageFeature{location: Point::new(x, y)}
    }
}

impl Feature for ImageFeature {
    fn get_x_image_float(&self) -> Float { self.location.x }
    fn get_y_image_float(&self) -> Float {self.location.y}
    fn get_x_image(&self) -> usize { self.location.x.trunc() as usize}
    fn get_y_image(&self) -> usize { self.location.y.trunc() as usize}
    fn get_closest_sigma_level(&self) -> usize { 0}
}



#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Match<T : Feature> {
    pub feature_one: T,
    pub feature_two: T
}

/**
 * Conditions Matches on a Sphere of Radius 1.
 * TODO: Conditioning on a Cube of Side Length 1
 */
pub fn condition_matches<T: Feature>(matches: &Vec<Match<T>>) -> Vec<Match<ImageFeature>> {
    let number_of_matches = matches.len();
    let number_of_matches_float = matches.len() as Float;
    let mut image_coordiantes_one_as_matrix = OMatrix::<Float,U3,Dynamic>::zeros(number_of_matches);
    let mut image_coordiantes_two_as_matrix = OMatrix::<Float,U3,Dynamic>::zeros(number_of_matches);

    let mut one_x_acc = 0.0;
    let mut one_y_acc = 0.0;
    let mut two_x_acc = 0.0;
    let mut two_y_acc = 0.0;

    for c in 0..number_of_matches {
        let feature_one = &matches[c].feature_one;
        let feature_two = &matches[c].feature_two;

        let image_one_coords = feature_one.get_as_3d_point(1.0);
        let image_two_coords = feature_two.get_as_3d_point(1.0);

        image_coordiantes_one_as_matrix.column_mut(c).copy_from(&image_one_coords);
        image_coordiantes_two_as_matrix.column_mut(c).copy_from(&image_two_coords);

        one_x_acc+=image_one_coords[0];
        one_y_acc+=image_one_coords[1];
        two_x_acc+=image_two_coords[0];
        two_y_acc+=image_two_coords[1];

    }

    let one_x_center  = one_x_acc/ number_of_matches_float;
    let one_y_center  = one_y_acc/ number_of_matches_float;
    let two_x_center  = two_x_acc/ number_of_matches_float;
    let two_y_center  = two_y_acc/ number_of_matches_float;

    let transform_one_dist = Matrix3::<Float>::new(1.0,0.0,-one_x_center,
                                                   0.0,1.0,-one_y_center,
                                                   0.0,0.0,1.0);
    let transform_two_dist = Matrix3::<Float>::new(1.0,0.0,-two_x_center,
                                                   0.0,1.0,-two_y_center,
                                                   0.0,0.0,1.0);

    //Transform points so that centroid is at the origin                                   
    let centered_features_one = transform_one_dist*image_coordiantes_one_as_matrix;
    let centered_features_two = transform_two_dist*image_coordiantes_two_as_matrix;
    

    let mut avg_distance_one = 0.0;
    let mut avg_distance_two = 0.0;
    for c in 0..number_of_matches {

        let c_one = centered_features_one.column(c);
        let c_two = centered_features_two.column(c);

        avg_distance_one += (c_one[0].powi(2)+c_one[1].powi(2)).sqrt();
        avg_distance_two += (c_two[0].powi(2)+c_two[1].powi(2)).sqrt();
    }

    let sqrt_two = (2.0 as Float).sqrt();
    let scale_one = number_of_matches_float*sqrt_two/avg_distance_one;
    let scale_two = number_of_matches_float*sqrt_two/avg_distance_two;

    let mut conditioned_matches = Vec::<Match<ImageFeature>>::with_capacity(number_of_matches);

    //Scale so that the average distance from the origin is sqrt(2)
    for c in 0..number_of_matches {
        let c_one = centered_features_one.column(c);
        let c_two = centered_features_two.column(c);

        let m = Match { feature_one: ImageFeature::new(c_one[0]*scale_one,c_one[1]*scale_one), feature_two: ImageFeature::new(c_two[0]*scale_two,c_two[1]*scale_two)};
        conditioned_matches.push(m);
    }
    conditioned_matches
}
