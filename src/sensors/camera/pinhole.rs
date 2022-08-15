extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use na::{convert, U1,U3, Matrix2x3,Matrix3, Vector, Vector3, base::storage::Storage, SimdRealField, ComplexField,base::Scalar};
use num_traits::{float,NumAssign};
use simba::scalar::{SubsetOf,SupersetOf};
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::Camera; 

#[derive(Copy,Clone)]
pub struct Pinhole<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField> {
    pub projection: Matrix3<F>,
    pub inverse_projection: Matrix3<F>
}

impl<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField> Pinhole<F> {
    pub fn new(fx: F, fy: F, cx: F, cy: F, invert_focal_length: bool) -> Pinhole<F> {
       let factor = match invert_focal_length {
           true => -F::one(),
           false => F::one()
       };
       let fx_scaled = factor*fx;
       let fy_scaled = factor*fy;
       let projection = Matrix3::<F>::new(
        fx_scaled, F::zero(), cx,
       F::zero(), fy_scaled, cy,
       F::zero(), F::zero(), F::one());
       let inverse_projection = Matrix3::<F>::new(
        F::one()/fx_scaled,F::zero(), -cx/fx_scaled,
       F::zero(),F::one()/fy_scaled, -cy/fy_scaled,
       F::zero(), F::zero(), F::one());

      Pinhole{projection,inverse_projection}
    }


    pub fn from_matrix(mat: &Matrix3<F>, invert_focal_length: bool) -> Pinhole<F> {
        Pinhole::new(mat[(0,0)],mat[(1,1)],mat[(0,2)],mat[(1,2)],invert_focal_length)
    }

    pub fn get_fx(&self) -> F {
        self.projection[(0,0)]
    }

    pub fn get_fy(&self) -> F {
        self.projection[(1,1)]
    }

    pub fn get_cx(&self) -> F {
        self.projection[(0,2)]
    }

    pub fn get_cy(&self) -> F {
        self.projection[(1,2)]
    }

    pub fn cast<F2: num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField + SubsetOf<F> + SupersetOf<F>>(&self) -> Pinhole<F2> {
        Pinhole::<F2>::new(convert(self.get_fx()),convert(self.get_fy()),convert(self.get_cx()),convert(self.get_cy()),false)
    }
}

impl<F: float::Float + Scalar + NumAssign + SimdRealField + ComplexField> Camera<F> for Pinhole<F> {
    fn get_projection(&self) -> Matrix3<F> {
        self.projection
    }

    fn get_inverse_projection(&self) -> Matrix3<F> {
        self.inverse_projection
    }

    fn get_jacobian_with_respect_to_position_in_camera_frame<T>(&self, position: &Vector<F,U3,T>) -> Matrix2x3<F> where T: Storage<F,U3,U1> {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        let z_sqrd = float::Float::powi(z,2);

        Matrix2x3::<F>::new(self.get_fx()/z, F::zero() , -(self.get_fx()*x)/z_sqrd,
                                F::zero(), self.get_fy()/z,  -(self.get_fy()*y)/z_sqrd)

    }

    fn project<T>(&self, position: &Vector<F,U3,T>) -> Point<F> where T: Storage<F,U3,U1> {
        let z = position[2];
        let homogeneous = position/z;
        let projected_coordiantes = self.projection*homogeneous;
        Point::<F>::new(projected_coordiantes[0],projected_coordiantes[1])
    }

    fn backproject(&self, point: &Point<F>, depth: F) -> Vector3<F> {
        let homogeneous = Vector3::<F>::new(point.x, point.y,F::one());
        (self.inverse_projection*homogeneous).scale(depth)
    }

    fn get_focal_x(&self) -> F {
        self.get_fx()
    }

    fn get_focal_y(&self) -> F {
        self.get_fy()
    }
}



