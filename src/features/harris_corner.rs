extern crate nalgebra as na;

use na::Matrix2;
use crate::image::Image;
use crate::{Float, GradientDirection};
use crate::filter::{gradient_convolution_at_sample,prewitt_kernel::PrewittKernel};
use crate::features::Feature;


pub fn harris_matrix(images: &Vec<Image>, feature: &dyn Feature) -> Matrix2<Float> {

    let first_order_kernel = PrewittKernel::new();

    let dx = gradient_convolution_at_sample(images,feature,&first_order_kernel,GradientDirection::HORIZINTAL);
    let dy = gradient_convolution_at_sample(images,feature,&first_order_kernel,GradientDirection::VERTICAL);

    Matrix2::new(dx.powi(2),dx*dy,
                dx*dy,dy.powi(2))

}

pub fn harris_response(harris_matrix: &Matrix2<Float>, k: Float) -> Float {
    let determinant = harris_matrix.determinant();
    let trace = harris_matrix.trace();
    determinant - k*trace.powi(2)
}

pub fn harris_response_for_feature(images: &Vec<Image>, feature: &dyn Feature,  k: Float) -> Float {
    let harris_matrix = harris_matrix(images,feature);
    harris_response(&harris_matrix, k)
}



