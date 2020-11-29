extern crate nalgebra as na;

use na::{U1,U3,Vector,Vector3,Matrix2x3,Matrix3, base::storage::Storage};
use crate::Float;
use crate::features::geometry::point::Point;

pub mod pinhole;

pub trait Camera {
    fn get_projection(&self) -> Matrix3<Float>;
    fn get_inverse_projection(&self) -> Matrix3<Float>;
    fn get_jacobian_with_respect_to_position<T>(&self, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1>;
    fn project<T>(&self, position: &Vector<Float,U3,T>) -> Point<Float> where T: Storage<Float,U3,U1>;
    fn backproject(&self, point: &Point<Float>, depth: Float) -> Vector3<Float>;
}