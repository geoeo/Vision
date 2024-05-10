use na::{Vector3, Vector6, Isometry3, SMatrix, Point3, Matrix3x6};
use crate::numerics::{lie::exp_se3,pose::from_matrix};
use crate::sfm::state::cam_state::CamState;
use crate::sensors::camera::Camera;
use crate::GenericFloat;

/**
 * Format (u,w) where u is translation and w is rotation 
 */
pub const CAMERA_PARAM_SIZE: usize = 6; 

#[derive(Clone, Copy)]
pub struct CameraExtrinsicState<F: GenericFloat, C: Camera<F> + Clone> {
    state: Isometry3<F>,
    camera: C
}

impl<F: GenericFloat, C: Camera<F> + Clone> CamState<F,C,CAMERA_PARAM_SIZE> for CameraExtrinsicState<F,C> {
    fn new(raw_state: Vector6<F>, camera: &C) -> CameraExtrinsicState<F,C> {
        let translation = Vector3::<F>::new(raw_state[0], raw_state[1], raw_state[2]);
        let axis_angle = Vector3::<F>::new(raw_state[3],raw_state[4],raw_state[5]);
        let state = Isometry3::new(translation, axis_angle);
        CameraExtrinsicState{state, camera: camera.clone()}
    }

    fn update(&mut self, perturb: Vector6<F>) -> () {
        let u = perturb.fixed_rows::<3>(0);
        let w = perturb.fixed_rows::<3>(3);
        let delta_transform = exp_se3(&u, &w);
        
        let current_transform = self.state.to_matrix();
        let new_transform = delta_transform*current_transform;
        let new_isometry = from_matrix(&new_transform);
        self.state = new_isometry;
    }

    fn to_serial(&self) ->  [F; CAMERA_PARAM_SIZE] {
        let u = self.state.translation;
        let w = self.state.rotation.scaled_axis();
        [
            u.x,
            u.y,
            u.z,
            w.x,
            w.y,
            w.z,
        ]
    }

    fn get_position(&self) -> Isometry3<F> {
        self.state
    }

    fn duplicate(&self) -> Self {
        self.clone() 
    }

    fn get_camera(&self) -> C {
        self.camera.clone()
    }

    fn get_jacobian(&self, point: &Point3<F>, lie_jacobian: &Matrix3x6<F>) -> SMatrix<F,2,CAMERA_PARAM_SIZE> {
        self.camera.get_jacobian_with_respect_to_position_in_camera_frame(&point.coords).expect("Could not compute jacobian for camera state")*lie_jacobian
    }

    
}