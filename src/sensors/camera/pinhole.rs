extern crate nalgebra as na;

use na::{U1,U3, Matrix2x3,Matrix3,Matrix4, Vector, Vector3,Vector4, base::storage::Storage};
use crate::Float;
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::Camera; 

#[derive(Copy,Clone)]
pub struct Pinhole {
    pub projection: Matrix3<Float>,
    pub inverse_projection: Matrix3<Float>
}

impl Pinhole {
    pub fn new(fx: Float, fy: Float, cx: Float, cy: Float, invert_focal_length: bool) -> Pinhole {
       let factor = match invert_focal_length {
           true => -1.0,
           false => 1.0
       };
       let fx_scaled = factor*fx;
       let fy_scaled = factor*fy;
       let projection = Matrix3::<Float>::new(fx_scaled, 0.0, cx,
                                              0.0, fy_scaled, cy,
                                              0.0, 0.0, 1.0);
       let inverse_projection = Matrix3::<Float>::new(1.0/fx_scaled,0.0, -cx/fx_scaled,
                                                      0.0,1.0/fy_scaled, -cy/fy_scaled,
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

    fn get_jacobian_with_respect_to_position_in_camera_frame<T>(&self, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1> {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        let z_sqrd = z.powi(2);

        Matrix2x3::<Float>::new(self.get_fx()/z, 0.0 , -self.get_fx()*x/z_sqrd,
                                0.0, self.get_fy()/z,  -self.get_fy()*y/z_sqrd)

    }

    fn get_jacobian_with_respect_to_position_in_world_frame<T>(&self, transformation: &Matrix4<Float>, position: &Vector<Float,U3,T>) -> Matrix2x3<Float> where T: Storage<Float,U3,U1> {
        let homogeneous_point_cam = transformation*Vector4::<Float>::new(position[0],position[1],position[2],1.0);
        let x_c = homogeneous_point_cam[0];
        let y_c = homogeneous_point_cam[1];
        let z_c = homogeneous_point_cam[2];
        let z_c_sqrd = z_c.powi(2);
        let r_xx = transformation[(0,0)];
        let r_yx = transformation[(1,0)];
        let r_zx = transformation[(2,0)];
        let r_xy =  transformation[(0,1)];
        let r_yy =  transformation[(1,1)];
        let r_zy =  transformation[(2,1)];
        let r_xz =  transformation[(0,2)];
        let r_yz =  transformation[(1,2)];
        let r_zz =  transformation[(2,2)];

        let mut jacobian = Matrix2x3::<Float>::zeros();

        jacobian[(0,0)] = ((self.get_fx()*r_xx + self.get_cx()*r_zx)*z_c - r_zx*(self.get_fx()*x_c+self.get_cx()*z_c)) / z_c_sqrd;
        jacobian[(1,0)] = ((self.get_fy()*r_yx + self.get_cy()*r_zx)*z_c - r_zx*(self.get_fy()*y_c+self.get_cy()*z_c)) / z_c_sqrd;
        
        jacobian[(0,1)] = ((self.get_fx()*r_xy + self.get_cx()*r_zy)*z_c - r_zy*(self.get_fx()*x_c+self.get_cx()*z_c)) / z_c_sqrd;
        jacobian[(1,1)] = ((self.get_fx()*r_yy + self.get_cy()*r_zy)*z_c - r_zy*(self.get_fy()*y_c+self.get_cy()*z_c)) / z_c_sqrd;

        jacobian[(0,2)] = ((self.get_fx()*r_xz + self.get_cx()*r_zz)*z_c - r_zz*(self.get_fx()*x_c+self.get_cx()*z_c)) / z_c_sqrd;
        jacobian[(1,2)] = ((self.get_fy()*r_yz + self.get_cy()*r_zz)*z_c - r_zz*(self.get_fy()*y_c+self.get_cy()*z_c)) / z_c_sqrd;

        jacobian
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



