use na::{Vector3, Vector6, Isometry3};
use crate::numerics::{lie::exp_se3,pose::from_matrix};
use crate::GenericFloat;

/**
 * Format (u,w) where u is translation and w is rotation 
 */
pub const CAMERA_PARAM_SIZE: usize = 6; 

#[derive(Clone, Copy)]
pub struct CameraExtrinsicState<F: GenericFloat> {
    state: Isometry3<F>
}

impl<F: GenericFloat> CameraExtrinsicState<F> {
    pub fn new(raw_state: Vector6<F>) -> CameraExtrinsicState<F> {
        let translation = Vector3::<F>::new(raw_state[0], raw_state[1], raw_state[2]);
        let axis_angle = Vector3::<F>::new(raw_state[3],raw_state[4],raw_state[5]);
        let state = Isometry3::new(translation, axis_angle);
        CameraExtrinsicState{state}
    }

    pub fn update(&mut self, perturb: Vector6<F>) -> () {
        let u = perturb.fixed_rows::<3>(0);
        let w = perturb.fixed_rows::<3>(3);
        let delta_transform = exp_se3(&u, &w);
        
        let current_transform = self.state.to_matrix();
        let new_transform = delta_transform*current_transform;
        let new_isometry = from_matrix(&new_transform);
        self.state = new_isometry;
    }

    pub fn to_serial(&self) ->  [F; CAMERA_PARAM_SIZE] {
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

    pub fn get_position(&self) -> Isometry3<F> {
        self.state
    }
}