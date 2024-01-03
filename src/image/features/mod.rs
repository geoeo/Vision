use nalgebra as na;

use na::{Vector2,Vector3,Matrix3};
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
pub mod matches;
pub mod image_feature;


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

    fn get_as_homogeneous(&self, scaling: Float) -> Vector3<Float> {
       Vector3::<Float>::new(self.get_x_image_float()/scaling, self.get_y_image_float()/scaling, 1.0)
    }
    fn get_as_2d_point(&self) -> Vector2<Float> {
        Vector2::<Float>::new(self.get_x_image_float(), self.get_y_image_float())
    }
    /**
     * Gets the camera ray for image points
     */
    fn get_camera_ray(&self, inverse_intrinsics: &Matrix3<Float>) -> Vector3<Float> {
        inverse_intrinsics*Vector3::<Float>::new(self.get_x_image_float(), self.get_y_image_float(),1.0) 
    }

    /**
     * Gets the camera ray for image points which are defined in the coordiante system by Photogrammetic Computer Vision 
     */
    fn get_camera_ray_photogrammetric(&self, inverse_intrinsics: &Matrix3<Float>) -> Vector3<Float> {
        let ray = self.get_camera_ray(inverse_intrinsics);
        Vector3::<Float>::new(ray[0],-ray[1],-ray[2])
    }
    fn apply_normalisation(&self, norm: &Matrix3<Float>, depth: Float) -> Self;
    fn get_landmark_id(&self) -> Option<usize>;
    fn copy_with_landmark_id(&self, landmark_id: Option<usize>) -> Self;
    fn new(x: Float, y: Float, landmark_id: Option<usize>) -> Self;
    fn get_location(&self) -> Point<Float>;
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

pub fn compute_linear_normalization<Feat: Feature>(features: &Vec<Feat>) -> (Matrix3<Float>, Matrix3<Float>) {
    let l = features.len();
    let l_as_float = l as Float;

    let mut normalization_matrix = Matrix3::<Float>::identity();
    let mut normalization_matrix_inv= Matrix3::<Float>::identity();

    let mut avg = Vector2::<Float>::zeros();

    for feat in features{
        let f = feat.get_as_2d_point();
        avg[0] += f[0];
        avg[1] += f[1];
    }

    avg /= l_as_float;

    let dist_mean_norm =  features.iter().fold(0.0, |acc, f| (acc + (f.get_as_2d_point()-avg).norm_squared()));

    let sqrt_2 = (2.0 as Float).sqrt();
    let s = (sqrt_2*l_as_float)/dist_mean_norm.sqrt();

    normalization_matrix[(0,0)] = s;
    normalization_matrix[(1,1)] = s;
    normalization_matrix[(0,2)] = -s*avg[0];
    normalization_matrix[(1,2)] = -s*avg[1];

    normalization_matrix_inv[(0,0)] = 1.0/s;
    normalization_matrix_inv[(1,1)] = 1.0/s;
    normalization_matrix_inv[(0,2)] = avg[0];
    normalization_matrix_inv[(1,2)] = avg[1];


    (normalization_matrix, normalization_matrix_inv)

}


