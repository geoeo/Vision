extern crate nalgebra as na;

use na::Matrix2;
use crate::image::Image;
use crate::{Float, GradientDirection};
use crate::image::filter::{gradient_convolution_at_sample,prewitt_kernel::PrewittKernel,laplace_kernel::LaplaceKernel};
use crate::image::features::Feature;

pub fn hessian_matrix<F: Feature>(images: &Vec<Image>,x_gradients: &Vec<Image>, feature: &F) -> Matrix2<Float> {

    let second_order_kernel = LaplaceKernel::new();
    let first_order_kernel = PrewittKernel::new();

    let dxx = gradient_convolution_at_sample(images,feature,&second_order_kernel,GradientDirection::HORIZINTAL);
    let dyy = gradient_convolution_at_sample(images,feature,&second_order_kernel,GradientDirection::VERTICAL);
    let dxy = gradient_convolution_at_sample(x_gradients,feature,&first_order_kernel,GradientDirection::VERTICAL);

    Matrix2::new(dxx,dxy,
                 dxy,dyy)

}

pub fn eigenvalue_ratio(harris_matrix: &Matrix2<Float>, r: Float) -> (Float,Float) {
    let trace = harris_matrix.trace();
    let determinant = harris_matrix.determinant();
    ( trace.powi(2)/determinant,(r+1.0).powi(2)/r)
}

pub fn reject_edge_response_filter<F: Feature>(images: &Vec<Image>,x_gradients: &Vec<Image>, feature: &F, r: Float) -> bool {
    let hessian = hessian_matrix(images,x_gradients,feature);
    reject_edge(&hessian, r)
}

pub fn accept_edge_response_filter<F: Feature>(images: &Vec<Image>,x_gradients: &Vec<Image>, feature: &F, r: Float) -> bool {
    let hessian = hessian_matrix(images,x_gradients,feature);
    accept_edge(&hessian, r)
}

fn reject_edge(harris_matrix: &Matrix2<Float>, r: Float) -> bool {
    let (eigenvalue_ratio, r_ratio) = eigenvalue_ratio(harris_matrix, r);
    eigenvalue_ratio < r_ratio as Float && eigenvalue_ratio > 0.0
}

fn accept_edge(harris_matrix: &Matrix2<Float>, r: Float) -> bool {
    let (eigenvalue_ratio, r_ratio) = eigenvalue_ratio(harris_matrix, r);
    eigenvalue_ratio >= r_ratio as Float && eigenvalue_ratio > 0.0
}



