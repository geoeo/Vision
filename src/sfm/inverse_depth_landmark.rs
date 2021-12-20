extern crate nalgebra as na;

use na::{Vector3,Vector6,Matrix3x6, Isometry3};
use crate::{Float,float};
use crate::image::features::geometry::point::Point;
use crate::sensors::camera::Camera;

pub struct InverseLandmark {
    state: Vector6<Float>,
    m : Vector3<Float>
}

impl InverseLandmark {
    fn new<C: Camera>(cam_to_world: &Isometry3<Float>, image_coords: &Point<Float>, camera: &C) -> InverseLandmark {
        let image_coords_homogeneous = Vector3::<Float>::new(image_coords.x,image_coords.y,-1.0);
        let h_c = camera.get_inverse_projection()*image_coords_homogeneous;
        assert!(h_c[0] >= 0.0 && h_c[1] >= 0.0);
        let h_w = cam_to_world.transform_vector(&h_c);
        let theta = h_w[0].atan2(h_w[2]);
        //We are not negating here because we will also not negate sin(phi)
        let phi = h_w[1].atan2((h_w[0].powi(2)+h_w[2].powi(2)).sqrt());
        let m = Vector3::<Float>::new(
            phi.cos()*theta.sin(),
            phi.sin(),
            phi.cos()*theta.cos()
        );
        let state = Vector6::<Float>::new(cam_to_world.translation.vector[0],cam_to_world.translation.vector[1],cam_to_world.translation.vector[2],theta,phi,float::MAX);
        InverseLandmark{state,m}
    }

    fn get_direction(&self) -> Vector3<Float> {
        self.m
    }

    fn get_inverse_depth(&self) -> Float {
        self.state[5]
    }

    fn get_observing_cam_world(&self) -> Vector3<Float> {
        self.state.fixed_rows::<3>(0).into_owned()
    }

    fn get_euclidean_representation(&self) -> Vector3<Float> {
        self.get_observing_cam_world()+(1.0/self.get_inverse_depth())*self.get_direction()
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<Float>) -> Vector3<Float> {
        let translation = other_cam_world.translation;
        let rotation = other_cam_world.rotation;

        rotation.inverse()*(self.get_inverse_depth()*(self.get_observing_cam_world() - translation.vector) + self.get_direction())
    }

    fn get_theta(&self) -> Float {
        self.state[3]
    }

    fn get_phi(&self) -> Float {
        self.state[4]
    }

    fn jacobian(&self, world_to_cam: &Isometry3<Float>) -> Matrix3x6<Float> {
        let mut jacobian = Matrix3x6::<Float>::zeros();
        let rotation_matrix = world_to_cam.rotation.to_rotation_matrix();
        let tranlsation_vector = world_to_cam.translation.vector;

        let j_xyz = self.get_inverse_depth()*rotation_matrix.matrix();
        let j_p = rotation_matrix*self.get_observing_cam_world()+tranlsation_vector;
        let j_theta = Vector3::<Float>::new(self.get_phi().cos()*self.get_theta().cos(),0.0,self.get_phi().cos()*(-self.get_theta().sin()));
        let j_phi = Vector3::<Float>::new((-self.get_phi().sin())*self.get_theta().sin(),self.get_phi().cos(),(-self.get_phi().sin())*self.get_theta().cos());

        jacobian.fixed_slice_mut::<3,3>(0,0).copy_from(&j_xyz);
        jacobian.fixed_slice_mut::<3,1>(0,3).copy_from(&j_theta);
        jacobian.fixed_slice_mut::<3,1>(0,4).copy_from(&j_phi);
        jacobian.fixed_slice_mut::<3,1>(0,5).copy_from(&j_p);
        jacobian
    }
}