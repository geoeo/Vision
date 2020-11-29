extern crate nalgebra as na;

use na::{U1,U3, Matrix2x3,Matrix3, Vector, Vector3, base::storage::Storage};
use crate::Float;
use crate::features::geometry::point::Point;
use crate::camera::Camera;

pub struct Pinhole {
    pub projection: Matrix3<Float>,
    pub inverse_projection: Matrix3<Float>
}

impl Pinhole {
    pub fn new(fx: Float, fy: Float, cx: Float, cy: Float) -> Pinhole {
       let projection = Matrix3::<Float>::new(fx, 0.0, cx,
                                              0.0, fy, cy,
                                              0.0, 0.0, 1.0);
       let inverse_projection = Matrix3::<Float>::new(1.0/fx,0.0, -cx/fx,
                                                      1.0/fy,0.0, -cy/fy,
                                                      0.0, 0.0, 1.0);

      Pinhole{projection,inverse_projection}
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
}

impl Camera for Pinhole {
    fn get_projection(&self) -> Matrix3<Float> {
        self.projection
    }

    fn get_inverse_projection(&self) -> Matrix3<Float> {
        self.inverse_projection
    }

    fn get_jacobian_with_respect_to_position<T>(&self, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1> {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        let z_sqrd = z.powi(2);
        Matrix2x3::<Float>::new(self.get_fx()/z, 0.0 , -self.get_fx()*x/z_sqrd,
                                self.get_fy()/z, 0.0,  -self.get_fy()*y/z_sqrd)

    }

    fn project<T>(&self, position: &Vector<Float,U3,T>) -> Point<Float> where T: Storage<Float,U3,U1> {
        let z = position[2];
        let homogeneous = position/z;
        let projected_coordiantes = self.projection*homogeneous;
        Point::<Float>::new(projected_coordiantes[0],projected_coordiantes[1])
    }

    fn backproject(&self, point: &Point<Float>, depth: Float) -> Vector3<Float> {
        let homogeneous = Vector3::<Float>::new(point.x, point.y,1.0);
        depth*self.inverse_projection*homogeneous
    }
}



