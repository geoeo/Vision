extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use na::{convert,U1,U3, Matrix2x3,Matrix3, Vector, Vector3, base::storage::Storage, SimdRealField,base::Scalar};
use num_traits::{float,NumAssign};
use simba::scalar::{SubsetOf,SupersetOf};
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::Camera; 

#[derive(Copy,Clone)]
pub struct Perspective<F: float::Float + Scalar + NumAssign + SimdRealField> {
    pub projection: Matrix3<F>,
    pub inverse_projection: Matrix3<F>
}

 //@TODO: unify principal distance into enum
impl<F: float::Float + Scalar + NumAssign + SimdRealField> Perspective<F> {
    pub fn new(fx: F, fy: F, cx: F, cy: F, s: F, invert_focal_length: bool) -> Perspective<F> {
       let factor = match invert_focal_length {
           true => -F::one(),
           false => F::one()
       };
       let fx_scaled = factor*fx;
       let fy_scaled = factor*fy;
       let cx_scaled = cx;
       let cy_scaled = cy;
       let projection = Matrix3::<F>::new(fx_scaled, s, cx_scaled,
                                              F::zero(), fy_scaled, cy_scaled,
                                              F::zero(),  F::zero(), F::one());
        

       let k = -cx_scaled/fx_scaled + s*cy_scaled*fx_scaled/fy_scaled;
       let inverse_projection = Matrix3::<F>::new(F::one()/fx_scaled, -s*fx_scaled/fy_scaled, k,
                                                  F::zero(), F::one()/fy_scaled, -cy_scaled/fy_scaled,
                                                  F::zero(), F::zero(), F::one());

        Perspective{projection,inverse_projection}
    }


    pub fn from_matrix(mat: &Matrix3<F>, invert_focal_length: bool) -> Perspective<F> {
        Perspective::new(mat[(0,0)],mat[(1,1)],mat[(0,2)],mat[(1,2)],mat[(0,1)],invert_focal_length)
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

    pub fn get_s(&self) -> F {
        self.projection[(0,1)]
    }

    pub fn cast<F2: num_traits::float::Float + Scalar + NumAssign + SimdRealField + SubsetOf<F> + SupersetOf<F>>(&self) -> Perspective<F2> {
        Perspective::<F2>::new(convert(self.get_fx()),convert(self.get_fy()),convert(self.get_cx()),convert(self.get_cy()),convert(self.get_s()),false)
    }
}

impl<F: float::Float + Scalar + NumAssign + SimdRealField> Camera<F> for Perspective<F> {

    fn from_matrices(projection: &Matrix3<F>, inverse_projection: &Matrix3<F>) -> Self {
        Perspective{projection: projection.clone(), inverse_projection: inverse_projection.clone()}
    }

    fn get_projection(&self) -> Matrix3<F> {
        self.projection
    }

    fn get_inverse_projection(&self) -> Matrix3<F> {
        self.inverse_projection
    }

    fn get_jacobian_with_respect_to_position_in_camera_frame<T, F2: float::Float + Scalar + SupersetOf<F>>(&self, position: &Vector<F2,U3,T>) -> Option<Matrix2x3<F2>> where T: Storage<F2,U3,U1> {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        match z {
            z if z.abs() > F2::zero() => {
                let z_sqrd = F2::powi(z,2);

                let fx = na::convert::<F,F2>(self.get_fx());
                let fy = na::convert::<F,F2>(self.get_fy());
                let s = na::convert::<F,F2>(self.get_s());
        
                Some(Matrix2x3::<F2>::new(fx/z, s/z , -(fx*x)/z_sqrd,
                                        F2::zero(), fy/z,  -(fy*y)/z_sqrd))
            },
            _ => None
        }
    }

    fn project<T, F2: float::Float + Scalar + SupersetOf<F> + SimdRealField>(&self, position: &Vector<F2,U3,T>) -> Option<Point<F2>> where T: Storage<F2,U3,U1> {
        let z = position[2];
        match z {
            z if z.abs() > F2::zero() => {
                let homogeneous = position/z;
                let proj = Matrix3::<F2>::from_iterator(self.projection.iter().map(|v| na::convert::<F,F2>(*v)));
                let projected_coordiantes = proj*homogeneous;
                Some(Point::<F2>::new(projected_coordiantes[0],projected_coordiantes[1]))
            },
            _ => None
        }

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
