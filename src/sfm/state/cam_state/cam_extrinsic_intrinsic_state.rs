use std::collections::HashMap;
use na::{Vector3, Vector5,SVector, Isometry3};
use crate::numerics::{lie::exp_se3,pose::from_matrix};
use crate::sfm::state::cam_state::CamState;
use crate::sensors::camera::{INTRINSICS,Camera};
use crate::GenericFloat;


/**
 * Format (u,w,fx,fy,cx,cy,s) where u is translation and w is rotation 
 */
pub const CAMERA_PARAM_SIZE: usize = 6 + 5; 

#[derive(Clone, Copy)]
pub struct CameraExtrinsicIntrinsicState<F: GenericFloat, C: Camera<F> + Clone> {
    extrinsic: Isometry3<F>,
    intrinsic: Vector5<F>,
    camera: C
}

impl<F: GenericFloat, C: Camera<F> + Clone> CamState<F, C, CAMERA_PARAM_SIZE> for CameraExtrinsicIntrinsicState<F,C> {
    fn new(raw_state: SVector<F,CAMERA_PARAM_SIZE>, camera: &C) -> CameraExtrinsicIntrinsicState<F,C> {
        let translation = Vector3::<F>::new(raw_state[0], raw_state[1], raw_state[2]);
        let axis_angle = Vector3::<F>::new(raw_state[3],raw_state[4],raw_state[5]);
        let extrinsic = Isometry3::new(translation, axis_angle);
        let intrinsic = Vector5::<F>::new(raw_state[6],raw_state[7],raw_state[8],raw_state[9],raw_state[10]);
        CameraExtrinsicIntrinsicState{extrinsic,intrinsic, camera: camera.clone()}
    }

    fn update(&mut self, perturb: SVector<F,CAMERA_PARAM_SIZE>) -> () {
        let u = perturb.fixed_rows::<3>(0);
        let w = perturb.fixed_rows::<3>(3);
        let delta_transform = exp_se3(&u, &w);
        
        let current_transform = self.extrinsic.to_matrix();
        let new_transform = delta_transform*current_transform;
        let new_isometry = from_matrix(&new_transform);
        self.extrinsic = new_isometry;

        let update_map = HashMap::<INTRINSICS,F>::from([(INTRINSICS::FX,perturb[6]),(INTRINSICS::FY,perturb[7]),(INTRINSICS::CX,perturb[8]),(INTRINSICS::CY,perturb[9]),(INTRINSICS::S,perturb[10])]);
        self.camera.update(&update_map);
    }

    fn to_serial(&self) ->  [F; CAMERA_PARAM_SIZE] {
        let u = self.extrinsic.translation;
        let w = self.extrinsic.rotation.scaled_axis();
        [
            u.x,
            u.y,
            u.z,
            w.x,
            w.y,
            w.z,
            self.intrinsic[0],
            self.intrinsic[1],
            self.intrinsic[2],
            self.intrinsic[3],
            self.intrinsic[4],
        ]
    }

    fn get_position(&self) -> Isometry3<F> {
        self.extrinsic
    }

    fn duplicate(&self) -> Self {
        self.clone() 
    }

    fn get_camera(&self) -> C {
        self.camera.clone()
    }
}