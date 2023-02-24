use nalgebra as na;

use na::{Vector2,Vector3,Matrix3};
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
pub mod feature_track;
pub mod solver_feature;


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
     * Gets the camera ray for image points which are assumed to be opposite of the principal plane 
     */
    fn get_camera_ray(&self, inverse_intrinsics: &Matrix3<Float>, positive_principal_distance: bool) -> Vector3<Float> {
        match positive_principal_distance {
            false => inverse_intrinsics*Vector3::<Float>::new(-self.get_x_image_float(), -self.get_y_image_float(),1.0),
            true => -1.0*inverse_intrinsics*Vector3::<Float>::new(self.get_x_image_float(), self.get_y_image_float(),1.0)
        }

    }
    fn apply_normalisation(&self, norm: &Matrix3<Float>, depth: Float) -> Self;
}


pub trait Oriented {
    fn get_orientation(&self) -> Float;
}

pub fn orientation<F: Feature>(source_images: &Vec<Image>, feature: &F) -> Float {
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

impl PartialEq for ImageFeature {
    fn eq(&self, other: &Self) -> bool {
        self.location == other.location
    }
}

impl Feature for ImageFeature {
    fn get_x_image_float(&self) -> Float { self.location.x }
    fn get_y_image_float(&self) -> Float { self.location.y }
    fn get_x_image(&self) -> usize { self.location.x.trunc() as usize}
    fn get_y_image(&self) -> usize { self.location.y.trunc() as usize}
    fn get_closest_sigma_level(&self) -> usize {0}
    fn apply_normalisation(&self, norm: &Matrix3<Float>, depth: Float) -> Self {
        let v = norm*self.get_as_3d_point(depth);
        ImageFeature::new(v[0], v[1])
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Match<T : Feature> {
    pub feature_one: T,
    pub feature_two: T,
    pub landmark_id: Option<usize>
}

impl<T: Feature> Match<T> {
    pub fn apply_normalisation(&self, norm_one: &Matrix3<Float>, norm_two: &Matrix3<Float>, depth: Float) -> Self {
        let feature_one = self.feature_one.apply_normalisation(norm_one, depth);
        let feature_two = self.feature_two.apply_normalisation(norm_two, depth);

        Match {feature_one, feature_two, landmark_id: None}
    }
}


impl<T: Feature + PartialEq> PartialEq for Match<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.feature_one == other.feature_one) && (self.feature_two == other.feature_two)
    }
}

//TODO
pub fn subsample_matches<T: Feature + Clone>(matches: Vec<Match<T>>, _: usize, _: usize) -> Vec<Match<T>> {
    matches
}