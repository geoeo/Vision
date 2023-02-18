extern crate nalgebra as na;
extern crate num_traits;

use na::{Vector3,Vector6,Matrix3x6, Isometry3, Point3,SVector, SMatrix, RealField,base::Scalar};
use num_traits::{float,NumAssign};

use crate::sfm::landmark::Landmark;
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::Camera;

#[derive(Copy,Clone)]
pub struct InverseLandmark<F: num_traits::float::Float + Scalar + NumAssign + RealField > {
    state: Vector6<F>,
    m : Vector3<F>
}

//We are not negating h_y because we will also not negate sin(phi)
impl<F: float::Float + Scalar + NumAssign + RealField> Landmark<F, 6> for InverseLandmark<F> {

    fn from_state(state: SVector<F,6>) -> InverseLandmark<F> {
        let theta = state[3];
        let phi = state[4];
        let m = InverseLandmark::direction(theta,phi);
        InverseLandmark{state,m}
    }

    fn from_array(arr: &[F; 6]) -> InverseLandmark<F> {
        let state = Vector6::<F>::new(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5]);
        let theta = arr[3];
        let phi = arr[4];
        let m = InverseLandmark::direction(theta,phi);
        InverseLandmark{state,m}
    }


    fn get_state_as_vector(&self) -> &Vector6<F>{
        &self.state
    }

    fn get_state_as_array(&self) -> [F; 6] {
        [self.state[0], self.state[1], self.state[2],self.state[3],self.state[4],self.state[5]]
    }

    fn get_euclidean_representation(&self) -> Point3<F> {
        self.get_observing_cam_in_world()+self.get_direction().scale(F::one()/self.get_inverse_depth())
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<F>) -> Point3<F> {
        let translation = other_cam_world.translation;
        let rotation = other_cam_world.rotation;

        let diff = self.get_observing_cam_in_world().coords - translation.vector;
        Point3::<F>::from(rotation.inverse()*(diff.scale(self.get_inverse_depth()) + self.get_direction()))
    }

    fn jacobian(&self, world_to_cam: &Isometry3<F>) -> SMatrix<F,3,6> {
        let mut jacobian = Matrix3x6::<F>::zeros();
        let rotation_matrix = world_to_cam.rotation.to_rotation_matrix();
        let translation_vector = world_to_cam.translation.vector;
        let observing_cam_in_world = self.get_observing_cam_in_world().coords;

        let j_xyz = rotation_matrix.matrix().scale(self.get_inverse_depth());
        let j_p = rotation_matrix*observing_cam_in_world+translation_vector;
        let j_theta = Vector3::<F>::new(float::Float::cos(self.get_phi())*float::Float::cos(self.get_theta()),F::zero(),float::Float::cos(self.get_phi())*(-float::Float::sin(self.get_theta())));
        let j_phi = Vector3::<F>::new((-float::Float::sin(self.get_phi()))*float::Float::sin(self.get_theta()),float::Float::cos(self.get_phi()),(-float::Float::sin(self.get_phi()))*float::Float::cos(self.get_theta()));

        jacobian.fixed_slice_mut::<3,3>(0,0).copy_from(&j_xyz);
        jacobian.fixed_slice_mut::<3,1>(0,3).copy_from(&j_theta);
        jacobian.fixed_slice_mut::<3,1>(0,4).copy_from(&j_phi);
        jacobian.fixed_slice_mut::<3,1>(0,5).copy_from(&j_p);
        jacobian
    }

    fn update(&mut self, perturb:&SVector<F,6>) -> () {
        self.state += perturb;
        let theta = self.get_theta();
        let phi = self.get_phi();
        self.m = InverseLandmark::direction(theta,phi);

    }
}

impl<F: num_traits::float::Float + Scalar + NumAssign + RealField> InverseLandmark<F> {

    pub fn new<C: Camera<F>>(cam_to_world: &Isometry3<F>, image_coords: &Point<F>, inverse_depth_prior: F, camera: &C) -> InverseLandmark<F> {
        assert!(inverse_depth_prior > F::zero()); // Negative depth is induced by image_coords_homogeneous
        let image_coords_homogeneous = Vector3::<F>::new(image_coords.x,image_coords.y, -F::one());
        let h_c = camera.get_inverse_projection()*image_coords_homogeneous;
        let h_w = cam_to_world.transform_vector(&h_c);
        let theta = float::Float::atan2(h_w[0],h_w[2]);
        //We are not negating h_w[1] here because we will also not negate sin(phi)
        let phi = float::Float::atan2(h_w[1],float::Float::sqrt(float::Float::powi(h_w[0],2)+float::Float::powi(h_w[2],2)));
        let m = InverseLandmark::direction(theta,phi);
        let state = Vector6::<F>::new(cam_to_world.translation.vector[0],cam_to_world.translation.vector[1],cam_to_world.translation.vector[2],theta,phi,inverse_depth_prior);
        InverseLandmark{state,m}
    }

    
    fn get_direction(&self) -> Vector3<F> {
        self.m
    }

    fn get_inverse_depth(&self) -> F {
        self.state[5]
    }

    fn get_observing_cam_in_world(&self) -> Point3<F> {
        Point3::<F>::new(self.state[0],self.state[1],self.state[2])
    }

    fn get_theta(&self) -> F {
        self.state[3]
    }

    fn get_phi(&self) -> F {
        self.state[4]
    }

    fn direction(theta: F, phi: F) -> Vector3<F> {
        Vector3::<F>::new(
            float::Float::cos(phi)*float::Float::sin(theta),
            float::Float::sin(phi),
            float::Float::cos(phi)*float::Float::cos(theta)
        )
    }

}