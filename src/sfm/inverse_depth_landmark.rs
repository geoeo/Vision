extern crate nalgebra as na;

use na::{Vector3,Vector6,Matrix3x6, Isometry3, Point3,SVector, SMatrix};
use crate::Float;
use crate::sfm::landmark::Landmark;
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::Camera;

#[derive(Copy,Clone)]
pub struct InverseLandmark {
    state: Vector6<Float>,
    m : Vector3<Float>
}

//We are not negating h_y because we will also not negate sin(phi)
impl Landmark<6> for InverseLandmark {

    fn from_state(state: SVector<Float,6>) -> InverseLandmark {
        let theta = state[3];
        let phi = state[4];
        let m = InverseLandmark::direction(theta,phi);
        InverseLandmark{state,m}
    }

    fn from_array(arr: &[Float; InverseLandmark::LANDMARK_PARAM_SIZE]) -> InverseLandmark {
        let state = Vector6::<Float>::new(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5]);
        let theta = arr[3];
        let phi = arr[4];
        let m = InverseLandmark::direction(theta,phi);
        InverseLandmark{state,m}
    }


    fn get_state_as_vector(&self) -> &Vector6<Float>{
        &self.state
    }

    fn get_state_as_array(&self) -> [Float; InverseLandmark::LANDMARK_PARAM_SIZE] {
        [self.state[0], self.state[1], self.state[2],self.state[3],self.state[4],self.state[5]]
    }

    fn get_euclidean_representation(&self) -> Point3<Float> {
        self.get_observing_cam_in_world()+(1.0/self.get_inverse_depth())*self.get_direction()
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<Float>) -> Point3<Float> {
        let translation = other_cam_world.translation;
        let rotation = other_cam_world.rotation;

        rotation.inverse()*(self.get_inverse_depth()*(self.get_observing_cam_in_world() - translation.vector) + self.get_direction())
    }

    fn jacobian(&self, world_to_cam: &Isometry3<Float>) -> SMatrix<Float,3,{Self::LANDMARK_PARAM_SIZE}> {
        let mut jacobian = Matrix3x6::<Float>::zeros();
        let rotation_matrix = world_to_cam.rotation.to_rotation_matrix();
        let translation_vector = world_to_cam.translation.vector;
        let observing_cam_in_world = self.get_observing_cam_in_world().coords;

        let j_xyz = self.get_inverse_depth()*rotation_matrix.matrix();
        let j_p = rotation_matrix*observing_cam_in_world+translation_vector;
        let j_theta = Vector3::<Float>::new(self.get_phi().cos()*self.get_theta().cos(),0.0,self.get_phi().cos()*(-self.get_theta().sin()));
        let j_phi = Vector3::<Float>::new((-self.get_phi().sin())*self.get_theta().sin(),self.get_phi().cos(),(-self.get_phi().sin())*self.get_theta().cos());

        jacobian.fixed_slice_mut::<3,3>(0,0).copy_from(&j_xyz);
        jacobian.fixed_slice_mut::<3,1>(0,3).copy_from(&j_theta);
        jacobian.fixed_slice_mut::<3,1>(0,4).copy_from(&j_phi);
        jacobian.fixed_slice_mut::<3,1>(0,5).copy_from(&j_p);
        jacobian
    }

    fn update(&mut self, perturb:&SVector<Float,6>) -> () {
        self.state += perturb;
        let theta = self.get_theta();
        let phi = self.get_phi();
        self.m = InverseLandmark::direction(theta,phi);

    }
}

impl InverseLandmark {

    pub fn new<C: Camera>(cam_to_world: &Isometry3<Float>, image_coords: &Point<Float>, inverse_depth_prior: Float, camera: &C) -> InverseLandmark {
        let image_coords_homogeneous = Vector3::<Float>::new(image_coords.x,image_coords.y, -1.0);
        let h_c = camera.get_inverse_projection()*image_coords_homogeneous;
        let h_w = cam_to_world.transform_vector(&h_c);
        let theta = h_w[0].atan2(h_w[2]);
        //We are not negating h_w[1] here because we will also not negate sin(phi)
        let phi = h_w[1].atan2((h_w[0].powi(2)+h_w[2].powi(2)).sqrt());
        let m = InverseLandmark::direction(theta,phi);
        let state = Vector6::<Float>::new(cam_to_world.translation.vector[0],cam_to_world.translation.vector[1],cam_to_world.translation.vector[2],theta,phi,inverse_depth_prior);
        InverseLandmark{state,m}
    }

    
    fn get_direction(&self) -> Vector3<Float> {
        self.m
    }

    fn get_inverse_depth(&self) -> Float {
        self.state[5]
    }

    fn get_observing_cam_in_world(&self) -> Point3<Float> {
        Point3::<Float>::new(self.state[0],self.state[1],self.state[2])
    }

    fn get_theta(&self) -> Float {
        self.state[3]
    }

    fn get_phi(&self) -> Float {
        self.state[4]
    }

    fn direction(theta: Float, phi: Float) -> Vector3<Float> {
        Vector3::<Float>::new(
            phi.cos()*theta.sin(),
            phi.sin(),
            phi.cos()*theta.cos()
        )
    }

}