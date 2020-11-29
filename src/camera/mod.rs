extern crate nalgebra as na;

use na::{U1,U3,Vector,Vector3,Matrix2x3,Matrix3, base::storage::Storage};
use crate::Float;
use crate::features::geometry::point::Point;

pub mod pinhole;

pub trait Camera {
    fn get_projection(&self) -> Matrix3<Float>;
    fn get_inverse_projection(&self) -> Matrix3<Float>;
    fn get_jacobian_with_respect_to_position(&self, position: &Vector3<Float>) -> Matrix2x3<Float>;
    fn project<T>(&self, position: &Vector<Float,U3,T>) -> Point<Float> where T: Storage<Float,U3,U1>;
    fn unproject(&self, point: &Point<Float>, depth: Float) -> Vector3<Float>;
}