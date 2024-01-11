extern crate nalgebra as na;
extern crate simba;
extern crate num_traits;

use na::{Vector3,Vector6,Matrix3x6, Isometry3, Point3,SVector, SMatrix,Matrix3};

use crate::image::features::Feature;
use crate::sfm::landmark::Landmark;
use crate::{GenericFloat,Float};

#[derive(Copy,Clone)]
/**
 * state: x, y, z, theta, phi, rho (inv depth)
 */
pub struct InverseLandmark<F: GenericFloat> {
    state: Vector6<F>,
    m : Vector3<F>,
    id: Option<usize>
}

impl<F: GenericFloat> Landmark<F, 6> for InverseLandmark<F> {

    fn from_state(state: SVector<F,6>) -> InverseLandmark<F> {
        let theta = state[3];
        let phi = state[4];
        let m = InverseLandmark::direction(theta,phi);
        InverseLandmark{state,m,id: None}
    }

    fn from_state_with_id(state: SVector<F,6>, id: &Option<usize>) -> InverseLandmark<F> {
        let theta = state[3];
        let phi = state[4];
        let m = InverseLandmark::direction(theta,phi);
        InverseLandmark{state,m,id: *id}
    }

    fn from_array(arr: &[F; 6]) -> InverseLandmark<F> {
        let state = Vector6::<F>::new(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5]);
        let theta = arr[3];
        let phi = arr[4];
        let m = InverseLandmark::direction(theta,phi);
        InverseLandmark{state,m,id: None}
    }

    fn get_state_as_vector(&self) -> &Vector6<F>{
        &self.state
    }

    fn get_state_as_array(&self) -> [F; 6] {
        [self.state[0], self.state[1], self.state[2],self.state[3],self.state[4],self.state[5]]
    }

    fn get_euclidean_representation(&self) -> Point3<F> {
        self.get_first_observing_cam_in_world()+self.get_direction().scale(F::one()/self.get_inverse_depth())
    }

    fn transform_into_other_camera_frame(&self, other_cam_world: &Isometry3<F>) -> InverseLandmark<F> {
        let translation = other_cam_world.translation.vector;
        let rotation = other_cam_world.rotation;

        let new_fist_observed_cam = rotation*self.get_first_observing_cam_in_world() +translation;
        let new_direction = rotation*self.get_direction();

        let phi = num_traits::Float::asin(new_direction[1]);
        let theta = num_traits::Float::asin(new_direction[0]/num_traits::Float::cos(phi));

        let state = Vector6::<F>::new(new_fist_observed_cam[0],new_fist_observed_cam[1],new_fist_observed_cam[2],theta,phi,self.get_inverse_depth());

        Self::from_state_with_id(state,&self.get_id())
    }

    fn jacobian(&self, world_to_cam: &Isometry3<F>) -> SMatrix<F,3,6> {
        let mut jacobian = Matrix3x6::<F>::zeros();
        let rotation_matrix = world_to_cam.rotation.to_rotation_matrix();
        let (sin_theta,cos_theta) = num_traits::Float::sin_cos(self.get_theta());
        let (sin_phi,cos_phi) = num_traits::Float::sin_cos(self.get_phi());
        let inverse_depth = self.get_inverse_depth();

        let j_theta = Vector3::<F>::new(cos_theta*cos_phi,F::zero(),cos_phi*-sin_theta)/inverse_depth;
        let j_phi = Vector3::<F>::new(-sin_phi*sin_theta,cos_phi,cos_theta*-sin_phi)/inverse_depth;
        let j_p = Vector3::<F>::new(-cos_phi*sin_theta,-sin_phi,-cos_theta*cos_phi)/num_traits::Float::powi(inverse_depth,2);

        jacobian.fixed_view_mut::<3,3>(0,0).copy_from(&Matrix3::<F>::identity()); //X,Y,Z
        jacobian.fixed_view_mut::<3,1>(0,3).copy_from(&j_theta);
        jacobian.fixed_view_mut::<3,1>(0,4).copy_from(&j_phi);
        jacobian.fixed_view_mut::<3,1>(0,5).copy_from(&j_p);
        rotation_matrix*jacobian
    }

    fn update(&mut self, perturb: &SVector<F,6>) -> () {
        self.state += perturb;
        let theta = self.get_theta();
        let phi = self.get_phi();
        self.m = Self::direction(theta,phi);
    }

    fn set_state(&mut self, state :&SVector<F,6>) -> () {
       self.state = state.clone();
       let theta = self.get_theta();
       let phi = self.get_phi();
       self.m = Self::direction(theta,phi);
    }

    fn get_id(&self) -> Option<usize> {self.id}

    fn duplicate(&self) -> Self {
        self.clone() 
    }

}

impl<F: GenericFloat> InverseLandmark<F> {
    pub fn new<Feat: Feature>(cam_to_world: &Isometry3<F>, feature: &Feat, inverse_depth_prior: F, inverse_projection: &Matrix3<Float>, id: &Option<usize>) -> InverseLandmark<F> {
        let camera_pos = cam_to_world.translation.vector;
        //TODO: make Feat trait generic
        let camera_ray_world = cam_to_world.rotation*feature.get_camera_ray(inverse_projection).cast::<F>();
        let h_x = camera_ray_world[0];
        let h_y = camera_ray_world[1];
        let h_z = camera_ray_world[2];
        let theta = num_traits::Float::atan2(h_x,h_z);
        let phi = num_traits::Float::atan2(h_y,num_traits::Float::sqrt(num_traits::Float::powi(h_x,2)+num_traits::Float::powi(h_z,2)));

        let state = Vector6::<F>::new(camera_pos[0],camera_pos[1],camera_pos[2],theta,phi,inverse_depth_prior);
        Self::from_state_with_id(state,id)
    }

    
    fn get_direction(&self) -> Vector3<F> {
        self.m
    }

    fn get_inverse_depth(&self) -> F {
        self.state[5]
    }

    /**
     * Gets the position of the first camera that observed the landmark in world coordinates
     */
    fn get_first_observing_cam_in_world(&self) -> Point3<F> {
        Point3::<F>::new(self.state[0],self.state[1],self.state[2])
    }

    fn get_theta(&self) -> F {
        self.state[3]
    }

    fn get_phi(&self) -> F {
        self.state[4]
    }

    fn direction(theta: F, phi: F) -> Vector3<F> {
        let (sin_theta,cos_theta) = num_traits::Float::sin_cos(theta);
        let (sin_phi,cos_phi) = num_traits::Float::sin_cos(phi);

        Vector3::<F>::new(
            cos_phi*sin_theta,
            sin_phi,
            cos_phi*cos_theta
        )
    }
}