extern crate nalgebra as na;

use na::{U1,U3,Vector,Vector3,Matrix2x3,Matrix3,Matrix4, base::storage::Storage};
use crate::Float;
use crate::image::features::geometry::point::Point;

pub mod pinhole;
pub mod camera_data_frame;

pub trait Camera {
    fn get_projection(&self) -> Matrix3<Float>;
    fn get_inverse_projection(&self) -> Matrix3<Float>;
    fn get_jacobian_with_respect_to_position_in_camera_frame<T>(&self, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1>;
    fn get_jacobian_with_respect_to_position_in_world_frame<T>(&self, transformation: &Matrix4<Float>, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1>;
    fn project<T>(&self, position: &Vector<Float,U3,T>) -> Point<Float> where T: Storage<Float,U3,U1>;
    fn backproject(&self, point: &Point<Float>, depth: Float) -> Vector3<Float>;
}