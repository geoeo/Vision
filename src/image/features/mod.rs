use crate::{Float,float};
use crate::image::Image;
use crate::image::filter::{prewitt_kernel::PrewittKernel,gradient_convolution_at_sample};
use crate::GradientDirection;

pub mod geometry;
pub mod sift_feature;
pub mod fast_feature;
pub mod harris_corner;
pub mod hessian_response;
pub mod orb_feature;
pub mod intensity_centroid;


pub trait Feature {
    fn get_x_image(&self) -> usize;
    fn get_y_image(&self) -> usize;
    fn get_closest_sigma_level(&self) -> usize;
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
