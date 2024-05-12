extern crate nalgebra as na;
extern crate num_traits;

use std::collections::HashMap;
use na::{convert,U1,U3, Matrix2x3,Matrix2x5,Matrix3, Vector, Vector3,Vector5,SMatrix, base::storage::Storage};
use simba::scalar::SupersetOf;
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::{INTRINSICS,Camera}; 
use crate::GenericFloat;

const IDENTITY_EPS: f32 = 1e-12f32;

#[derive(Copy,Clone)]
pub struct Perspective<F: GenericFloat> {
    pub projection: Matrix3<F>,
    pub inverse_projection: Matrix3<F>
}

 //@TODO: unify principal distance into enum, re-evaluate invert focal length
impl<F: GenericFloat> Perspective<F> {
    pub fn new(fx: F, fy: F, cx: F, cy: F, s: F, invert_focal_length: bool) -> Perspective<F> {
        let factor = match invert_focal_length {
            true => -F::one(),
            false => F::one()
        };

        let fx_scaled = factor*fx;
        let fy_scaled = factor*fy;
        let cx_scaled = cx;
        let cy_scaled = cy;

        let (projection,inverse_projection) = Self::compute_projections(fx_scaled,fy_scaled,cx_scaled,cy_scaled,s);
        assert!(num_traits::Float::abs((projection*inverse_projection).determinant())- F::one() <= F::from_f32(IDENTITY_EPS).expect("Converstion failed!"));
        Perspective{projection,inverse_projection}
    }

    fn compute_projections(fx: F,fy: F, cx: F, cy: F, s: F) -> (Matrix3<F>,Matrix3<F>) {
        let projection = Matrix3::<F>::new(fx, s, cx,
            F::zero(), fy, cy,
            F::zero(),  F::zero(), F::one());


        let k = -cx/fx + s*cy*fx/fy;
        let inverse_projection = Matrix3::<F>::new(F::one()/fx, -s*fx/fy, k,
                        F::zero(), F::one()/fy, -cy/fy,
                        F::zero(), F::zero(), F::one());

        (projection,inverse_projection)
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

    pub fn cast<F2: GenericFloat + SupersetOf<F>>(&self) -> Perspective<F2> {
        Perspective::<F2>::new(na::convert(self.get_fx()),convert(self.get_fy()),convert(self.get_cx()),convert(self.get_cy()),convert(self.get_s()),false)
    }
}

impl<F: GenericFloat> Camera<F> for Perspective<F> {

    fn from_matrices(projection: &Matrix3<F>, inverse_projection: &Matrix3<F>) -> Self {
        Perspective{projection: projection.clone(), inverse_projection: inverse_projection.clone()}
    }

    fn get_projection(&self) -> Matrix3<F> {
        self.projection
    }

    fn get_inverse_projection(&self) -> Matrix3<F> {
        self.inverse_projection
    }

    fn get_jacobian_with_respect_to_position_in_camera_frame<S>(&self, position: &Vector<F,U3,S>) -> Option<Matrix2x3<F>> where S: Storage<F,U3,U1> {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        match z {
            z if num_traits::Float::abs(z) > F::zero() => {
                let z_sqrd = num_traits::Float::powi(z,2);

                let fx = self.get_fx();
                let fy = self.get_fy();
                let s = self.get_s();
        
                Some(Matrix2x3::<F>::new(fx/z, s/z , -(fx*x)/z_sqrd,
                                        F::zero(), fy/z,  -(fy*y)/z_sqrd))
            },
            _ => None
        }
    }

    fn get_jacobian_with_respect_to_intrinsics<S>(&self, position: &Vector<F,U3,S>) -> Option<Matrix2x5<F>> where S: Storage<F,U3,U1> {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        match z {
            z if num_traits::Float::abs(z) > F::zero() => {
                Some(Matrix2x5::<F>::new(x/z,F::zero() , y , F::one(), F::zero(),
                    F::zero(), y/z, F::zero() ,F::zero(), F::one()))

            },
            _ => None
        }

    }

    fn project<S>(&self, position: &Vector<F,U3,S>) -> Option<Point<F>> where S: Storage<F,U3,U1> {
        let z = position[2];
        match z {
            z if num_traits::Float::abs(z) > F::zero() => {
                let homogeneous = position/z;
                let proj =self.get_projection();
                let projected_coordiantes = proj*homogeneous;
                Some(Point::<F>::new(projected_coordiantes[0],projected_coordiantes[1]))
            },
            _ => None
        }

    }

    fn get_full_jacobian<S>(&self, position: &Vector<F,U3,S>) -> Option<SMatrix<F,2,8>> where S: Storage<F,U3,U1> {
        let option_j_p = self.get_jacobian_with_respect_to_position_in_camera_frame(position);
        let option_j_i = self.get_jacobian_with_respect_to_intrinsics(position);

        match (option_j_p,option_j_i) {
            (Some(j_p),Some(j_i)) => {
                let mut j = SMatrix::<F,2,8>::zeros();
                j.fixed_view_mut::<2,3>(0,0).copy_from(&j_p);
                j.fixed_view_mut::<2,5>(0,3).copy_from(&j_i);
                Some(j)
            },
            _ => None
        }
    }

    fn backproject(&self, point: &Point<F>, depth: F) -> Vector3<F> {
        let homogeneous = Vector3::<F>::new(point.x, point.y,F::one());
        (self.inverse_projection*homogeneous).scale(depth)
    }

    fn get_intrinsics(&self) -> Vector5<F> {
        Vector5::<F>::new(self.get_fx(),self.get_fy(),self.get_cx(),self.get_cy(),self.get_s())
    }


    fn update(&mut self, perturb: &HashMap<INTRINSICS,F>) -> () {
        let delta_fx = perturb.get(&INTRINSICS::FX).expect("fx not found in perturb update");
        let delta_fy = perturb.get(&INTRINSICS::FY).expect("fy not found in perturb update");
        let delta_cx = perturb.get(&INTRINSICS::CX).expect("cx not found in perturb update");
        let delta_cy = perturb.get(&INTRINSICS::CY).expect("cy not found in perturb update");
        let delta_s = perturb.get(&INTRINSICS::S).expect("s not found in perturb update");

        let new_fx = self.get_fx() + *delta_fx;
        let new_fy = self.get_fy() + *delta_fy;
        let new_cx = self.get_cx() + *delta_cx;
        let new_cy = self.get_cy() + *delta_cy;
        let new_s = self.get_s() + *delta_s;

        let (projection,inverse_projection) = Self::compute_projections(new_fx,new_fy,new_cx,new_cy,new_s);
        self.projection = projection;
        self.inverse_projection = inverse_projection;
    }

}
