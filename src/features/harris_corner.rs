extern crate nalgebra as na;

use na::Matrix2;
use crate::image::Image;
use crate::{Float, GradientDirection};
use crate::filter::{gradient_convolution_at_sample,prewitt_kernel::PrewittKernel,laplace_kernel::LaplaceKernel};
use crate::features::Feature;


//TODO: @Investigate: maybe precomputing the gradient images is more efficient
pub fn harris_matrix(images: &Vec<Image>,x_gradients: &Vec<Image>, input_params: &dyn Feature) -> Matrix2<Float> {

    let second_order_kernel = LaplaceKernel::new();
    let first_order_kernel = PrewittKernel::new();

    let dxx = gradient_convolution_at_sample(images,input_params,&second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample(images,input_params,&second_order_kernel,GradientDirection::VERTICAL);
    let dxy = gradient_convolution_at_sample(x_gradients,input_params,&first_order_kernel,GradientDirection::VERTICAL);

    Matrix2::new(dxx,dxy,
                dxy,dyy)

}

pub fn harris_response(harris_matrix: &Matrix2<Float>, k: Float) -> Float {
    let determinant = harris_matrix.determinant();
    let trace = harris_matrix.trace();
    determinant - k*trace.powi(2)
}

pub fn harris_ratio(harris_matrix: &Matrix2<Float>, r: Float) -> (Float,Float) {
    let trace = harris_matrix.trace();
    let determinant = harris_matrix.determinant();
    ( trace.powi(2)/determinant,(r+1.0).powi(2)/r)
}

pub fn reject_edge(harris_matrix: &Matrix2<Float>, r: Float) -> bool {
    let (harris_ratio, r_ratio) = harris_ratio(harris_matrix, r);
    harris_ratio < r_ratio as Float && harris_ratio > 0.0
}

pub fn accept_edge(harris_matrix: &Matrix2<Float>, r: Float) -> bool {
    let (harris_ratio, r_ratio) = harris_ratio(harris_matrix, r);
    harris_ratio >= r_ratio as Float && harris_ratio > 0.0
}


pub fn reject_edge_response_filter(images: &Vec<Image>,x_gradients: &Vec<Image>, input_params: &dyn Feature, r: Float) -> bool {
    let hessian = harris_matrix(images,x_gradients,input_params);
    reject_edge(&hessian, r)
}

pub fn accept_edge_response_filter(images: &Vec<Image>,x_gradients: &Vec<Image>, input_params: &dyn Feature, r: Float) -> bool {
    let hessian = harris_matrix(images,x_gradients,input_params);
    accept_edge(&hessian, r)
}