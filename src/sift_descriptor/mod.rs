extern crate nalgebra as na;

use na::{Matrix1x2,Matrix2,Matrix3,Matrix3x1};
use crate::{float,Float,round};
use crate::image::Image;

pub mod orientation_histogram;
pub mod local_image_descriptor;
pub mod feature_vector;
pub mod keypoint;

pub const DESCRIPTOR_BINS: usize = 16;
pub const ORIENTATION_BINS: usize = 8;



pub fn rotation_matrix_2d_from_orientation(orientation: Float) -> Matrix2<Float> {

    Matrix2::new(orientation.cos(), -orientation.sin(),
                orientation.sin(), orientation.cos())

}

pub fn gradient_and_orientation(x_gradient: &Image, y_gradient: &Image, x: usize, y: usize) -> (Float,Float) {

    let x_diff = round(x_gradient.buffer.index((y,x)).clone(),5);
    let y_diff = round(y_gradient.buffer.index((y,x)).clone(),5);

    let gradient = (x_diff.powi(2) + y_diff.powi(2)).sqrt();
    let orientation = match  y_diff.atan2(x_diff.clone()) {
        angle if angle < 0.0 => 2.0*float::consts::PI + angle,
        angle => angle
    };

    (gradient,orientation)
}


