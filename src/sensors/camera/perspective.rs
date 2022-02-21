extern crate nalgebra as na;

use na::{U1,U3, Matrix2x3,Matrix3, Vector, Vector3, base::storage::Storage};
use crate::Float;
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::Camera; 

#[derive(Copy,Clone)]
pub struct Perspective {
    pub projection: Matrix3<Float>,
    pub inverse_projection: Matrix3<Float>
}

impl Perspective {
    pub fn new(fx: Float, fy: Float, cx: Float, cy: Float, s: Float, invert_focal_length: bool) -> Perspective {
       let factor = match invert_focal_length {
           true => -1.0,
           false => 1.0
       };
       let fx_scaled = factor*fx;
       let fy_scaled = factor*fy;
       let projection = Matrix3::<Float>::new(fx_scaled, s, cx,
                                              0.0, fy_scaled, cy,
                                              0.0, 0.0, 1.0);
        

    
        //TODO: check this
       let inverse_projection = match projection.try_inverse() {
           Some(v) => v,
           None => Matrix3::<Float>::zeros()
       };

        Perspective{projection,inverse_projection}
    }


    pub fn from_matrix(mat: &Matrix3<Float>, invert_focal_length: bool) -> Perspective {
        Perspective::new(mat[(0,0)],mat[(1,1)],mat[(0,2)],mat[(1,2)],mat[(0,1)],invert_focal_length)
    }

    pub fn get_fx(&self) -> Float {
        self.projection[(0,0)]
    }

    pub fn get_fy(&self) -> Float {
        self.projection[(1,1)]
    }

    pub fn get_cx(&self) -> Float {
        self.projection[(0,2)]
    }

    pub fn get_cy(&self) -> Float {
        self.projection[(1,2)]
    }

    pub fn get_s(&self) -> Float {
        self.projection[(0,1)]
    }
}

impl Camera for Perspective {
    fn get_projection(&self) -> Matrix3<Float> {
        self.projection
    }

    fn get_inverse_projection(&self) -> Matrix3<Float> {
        self.inverse_projection
    }

    fn get_jacobian_with_respect_to_position_in_camera_frame<T>(&self, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1> {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        let z_sqrd = z.powi(2);

        Matrix2x3::<Float>::new(self.get_fx()/z, self.get_s()/z , -(self.get_fx()*x)/z_sqrd,
                                0.0, self.get_fy()/z,  -(self.get_fy()*y)/z_sqrd)

    }

    fn project<T>(&self, position: &Vector<Float,U3,T>) -> Point<Float> where T: Storage<Float,U3,U1> {
        let z = position[2];
        let homogeneous = position/z;
        let projected_coordiantes = self.projection*homogeneous;
        Point::<Float>::new(projected_coordiantes[0],projected_coordiantes[1])
    }

    fn backproject(&self, point: &Point<Float>, depth: Float) -> Vector3<Float> {
        let homogeneous = Vector3::<Float>::new(point.x, point.y,1.0);
        depth*(self.inverse_projection*homogeneous)
    }
}
