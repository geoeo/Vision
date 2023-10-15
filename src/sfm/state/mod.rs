

extern crate nalgebra as na;
extern crate num_traits;

use na::{DVector, SMatrix, Matrix4, Vector3, Isometry3, Quaternion, Rotation3,Translation3, base::Scalar, RealField};
use num_traits::float;
use crate::numerics::{lie::exp_se3,pose::from_matrix};
use crate::sfm::landmark::Landmark;

pub mod ba_state_linearizer;
pub mod pnp_state_linearizer;

/**
 * Format (u,w) where u is translation and w is rotation 
 */
pub const CAMERA_PARAM_SIZE: usize = 6; 

/**
 * This is ordered [cam_1,cam_2,..,cam_n,point_1,point_2,...,point_m]
 * cam is parameterized by [u_1,u_2,u_3,w_1,w_2,w_3]
 * point is parameterized by [x,y,z]
 * */
pub struct State<F: Scalar, L: Landmark<F,T>, const T: usize> {
    camera_positions: Vec<Isometry3<F>>, //TOOD: make this vector of Isometry
    landmarks: Vec<L>,
    pub n_cams: usize,
    pub n_points: usize,
}

impl<F: float::Float + Scalar + RealField, L: Landmark<F,T> + Copy + Clone, const T: usize> Clone for State<F,L,T> {
    fn clone(&self) -> State<F,L,T> {
        State::<F,L,T> {
            camera_positions: self.camera_positions.clone(),
            landmarks: self.landmarks.clone(), 
            n_cams: self.n_cams, 
            n_points: self.n_points
        }
    }
}

impl<F: float::Float + Scalar + RealField, L: Landmark<F,T> + Copy + Clone, const T: usize> State<F,L,T> {
    pub fn new(camera_positions: DVector<F>, landmarks:  Vec<L>, n_cams: usize, n_points: usize) -> State<F,L,T> {
        let mut camera_iso = Vec::<Isometry3<F>>::with_capacity(n_cams);
        for i in 0..n_cams {
            let offset = CAMERA_PARAM_SIZE * i;
            let arr = camera_positions.fixed_rows::<CAMERA_PARAM_SIZE>(offset);
            let translation = Vector3::<F>::new(arr[0], arr[1], arr[2]);
            let axis_angle = Vector3::<F>::new(arr[3],arr[4],arr[5]);
            camera_iso.push(Isometry3::new(translation, axis_angle));
        }
        State{camera_positions: camera_iso, landmarks , n_cams, n_points}
    }

    pub fn get_landmarks(&self) -> &Vec<L> {
        &self.landmarks
    }

    pub fn get_camera_positions(&self) -> &Vec<Isometry3<F>> {
        &self.camera_positions
    }

    pub fn copy_from(&mut self, other: &State<F,L,T>) -> () {
        assert!(self.n_cams == other.n_cams);
        assert!(self.n_points == other.n_points);
        self.camera_positions.copy_from_slice(other.get_camera_positions());
        self.landmarks.copy_from_slice(other.get_landmarks());
    }

    pub fn update(&mut self, perturb: &DVector<F>) -> () {
        for i in 0..self.camera_positions.len() {
            let linear_idx = i*CAMERA_PARAM_SIZE;
            let u = perturb.fixed_rows::<3>(linear_idx);
            let w = perturb.fixed_rows::<3>(linear_idx + 3);
            let delta_transform = exp_se3(&u, &w);
            
            let current_transform = self.camera_positions[i].to_matrix();

            let new_transform = delta_transform*current_transform;
            let new_isometry = from_matrix(&new_transform);
            self.camera_positions[i] = new_isometry;            
        }
        
        let landmark_offset = self.camera_positions.len()*CAMERA_PARAM_SIZE;
        for i in (landmark_offset..perturb.len()).step_by(T) {
            let landmark_id = (i-landmark_offset)/T;
            self.landmarks[landmark_id].update(&perturb.fixed_view::<T,1>(i,0).into());
        }
    }

    pub fn jacobian_wrt_world_coordiantes(&self, point_index: usize, cam_idx: usize) ->  SMatrix<F,3,T> {
        let state_idx = cam_idx/CAMERA_PARAM_SIZE;
        let iso = self.camera_positions[state_idx];
        self.landmarks[point_index].jacobian(&iso)
    }

    /**
     * cam_idx is the index of a solver i.e. in CAMERA_PARAM_SIZE space
     */
    pub fn to_se3(&self, cam_idx: usize) -> Matrix4<F> {
        assert!(cam_idx < self.n_cams*CAMERA_PARAM_SIZE);
        let state_idx = cam_idx/CAMERA_PARAM_SIZE;
        self.camera_positions[state_idx].to_matrix()
    }

    pub fn as_matrix_point(&self) -> (Vec<Isometry3<F>>, Vec<Vector3<F>>) {
        (self.camera_positions.clone(), 
        self.landmarks.iter().map(|l| l.get_euclidean_representation().coords).collect::<Vec<_>>())
    }

    pub fn to_serial(&self) -> (Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; T]>) {
        let mut cam_serial = Vec::<[F; CAMERA_PARAM_SIZE]>::with_capacity(self.n_cams);
        let mut points_serial = Vec::<[F; T]>::with_capacity(self.n_points);

        for i in 0..self.n_cams {
            let linear_idx = i*CAMERA_PARAM_SIZE;
            let iso = self.camera_positions[i];
            let u = iso.translation;
            let w = iso.rotation.scaled_axis();
            let arr: [F; CAMERA_PARAM_SIZE] = [
                u.x,
                u.y,
                u.z,
                w.x,
                w.y,
                w.z,
            ];
            cam_serial.push(arr);
        }

        for i in 0..self.landmarks.len() {
            points_serial.push(self.landmarks[i].get_state_as_array());
        }

        (cam_serial, points_serial)
    }

    pub fn from_serial((cam_serial, points_serial): &(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; T]>)) -> State<F,L,T> {
        let mut camera_positions = Vec::<Isometry3<F>>::with_capacity(cam_serial.len());
        let mut landmarks = Vec::<L>::with_capacity(points_serial.len());

        for i in 0..cam_serial.len() {
            let arr = cam_serial[i];
            let translation = Vector3::<F>::new(arr[0], arr[1], arr[2]);
            let axis_angle = Vector3::<F>::new(arr[3],arr[4],arr[5]);
            camera_positions.push(Isometry3::new(translation, axis_angle));
        }

        for i in 0..points_serial.len() {
            landmarks.push(L::from_array(&points_serial[i]));
        }

        State {
            camera_positions,
            landmarks,
            n_cams: cam_serial.len(),
            n_points: points_serial.len()
        }
    }
}
