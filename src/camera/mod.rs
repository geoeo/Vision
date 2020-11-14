extern crate nalgebra as na;

use na::{Matrix2x3,Matrix3, Vector3};
use crate::Float;
use crate::features::geometry::point::Point;

pub mod pinhole;

pub trait Camera {
    fn get_projection(&self) -> Matrix3<Float>;
    fn get_inverse_projection(&self) -> Matrix3<Float>;
    fn get_jacobian_with_respect_to_position(&self, position: &Vector3<Float>) -> Matrix2x3<Float>;
    fn project(&self, position: &Vector3<Float>) -> Point<Float>;
    fn unproject(&self, point: &Point<Float>, depth: Float) -> Vector3<Float>;
}