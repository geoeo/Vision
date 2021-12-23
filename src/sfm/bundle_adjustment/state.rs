extern crate nalgebra as na;

use crate::numerics::lie::exp_se3;
use crate::sfm::euclidean_landmark::EuclideanLandmark;
use crate::Float;
use na::{DVector, Matrix3, Matrix4, Vector3, Isometry3, Rotation3};

/**
 * This is ordered [cam_1,cam_2,..,cam_n,point_1,point_2,...,point_m]
 * cam is parameterized by [u_1,u_2,u_3,w_1,w_2,w_3]
 * point is parameterized by [x,y,z]
 * */
#[derive(Clone)]
pub struct State {
    camera_positions: DVector<Float>, //TOOD: make this vector of Isometry
    landmarks: Vec<EuclideanLandmark>,
    pub n_cams: usize,
    pub n_points: usize,
}

impl State {


    //TODO make this configurable, jacobians should also be accessed from here
    pub const CAM_TRANSLATION_PARAM_SIZE: usize = 3;
    pub const CAM_ROTATION_PARAM_SIZE: usize = 3;
    pub const CAM_PARAM_SIZE: usize = State::CAM_TRANSLATION_PARAM_SIZE + State::CAM_ROTATION_PARAM_SIZE;
    pub const LANDMARK_PARAM_SIZE: usize = EuclideanLandmark::LANDMARK_PARAM_SIZE;

    pub fn new(camera_positions: DVector<Float>, landmarks:  Vec<EuclideanLandmark>, n_cams: usize, n_points: usize) -> State {
        State{camera_positions, landmarks , n_cams, n_points}
    }

    pub fn get_landmarks(&self) -> &Vec<EuclideanLandmark> {
        &self.landmarks
    }

    pub fn get_camera_positions(&self) -> &DVector<Float> {
        &self.camera_positions
    }

    pub fn copy_from(&mut self, other: &State) -> () {
        assert!(self.n_cams == other.n_cams);
        assert!(self.n_points == other.n_points);
        self.camera_positions.copy_from(other.get_camera_positions());
        self.landmarks.copy_from_slice(other.get_landmarks());
    }

    pub fn update(&mut self, perturb: &DVector<Float>) -> () {


        for i in (0..self.camera_positions.nrows()).step_by(State::CAM_PARAM_SIZE) {
            let u = 1.0*perturb.fixed_rows::<{State::CAM_TRANSLATION_PARAM_SIZE}>(i);
            let w = 1.0*perturb.fixed_rows::<{State::CAM_ROTATION_PARAM_SIZE}>(i + State::CAM_TRANSLATION_PARAM_SIZE);
            let delta_transform = exp_se3(&u, &w);
            
            let current_transform = self.to_se3(i);

            let new_transform = delta_transform*current_transform;

            let new_translation = new_transform.fixed_slice::<{State::CAM_TRANSLATION_PARAM_SIZE},1>(0,State::CAM_TRANSLATION_PARAM_SIZE);
            self.camera_positions.fixed_slice_mut::<{State::CAM_TRANSLATION_PARAM_SIZE},1>(i,0).copy_from(&new_translation);

            let new_rotation = na::Rotation3::from_matrix(&new_transform.fixed_slice::<3,3>(0,0).into_owned());
            self.camera_positions.fixed_slice_mut::<{State::CAM_ROTATION_PARAM_SIZE},1>(i+State::CAM_TRANSLATION_PARAM_SIZE,0).copy_from(&(new_rotation.scaled_axis()));
        }

        let landmark_offset = self.camera_positions.nrows();
        for i in 0..self.landmarks.len() {
            let pertub_offset = i*State::LANDMARK_PARAM_SIZE;
            self.landmarks[i].update(perturb[landmark_offset+pertub_offset],perturb[landmark_offset+pertub_offset+1],perturb[landmark_offset+pertub_offset+2]);
        }
    }

    pub fn jacobian_wrt_world_coordiantes(&self, point_index: usize, cam_idx: usize) -> Matrix3<Float> {

        let translation = na::Vector3::new(self.camera_positions[cam_idx],self.camera_positions[cam_idx+1],self.camera_positions[cam_idx+2]);
        let axis_angle = na::Vector3::new(self.camera_positions[cam_idx+3],self.camera_positions[cam_idx+4],self.camera_positions[cam_idx+5]);
        let isometry = Isometry3::new(translation, axis_angle);
        self.landmarks[point_index].jacobian(&isometry)
    }

    pub fn to_se3(&self, cam_idx: usize) -> Matrix4<Float> {
        assert!(cam_idx < self.n_cams*State::CAM_PARAM_SIZE);
        let translation = self.camera_positions.fixed_rows::<{State::CAM_TRANSLATION_PARAM_SIZE}>(cam_idx);
        let axis = na::Vector3::new(self.camera_positions[cam_idx+3],self.camera_positions[cam_idx+4],self.camera_positions[cam_idx+5]);
        let axis_angle = Rotation3::new(axis);
        let mut transform = Matrix4::<Float>::identity();
        transform.fixed_slice_mut::<3,3>(0,0).copy_from(axis_angle.matrix());
        transform.fixed_slice_mut::<{State::CAM_TRANSLATION_PARAM_SIZE},1>(0,State::CAM_TRANSLATION_PARAM_SIZE).copy_from(&translation);
        transform
    }

    pub fn as_matrix_point(&self) -> (Vec<Isometry3<Float>>, Vec<Vector3<Float>>) {
        let mut cam_positions = Vec::<Isometry3<Float>>::with_capacity(self.n_cams);
        let mut points = Vec::<Vector3<Float>>::with_capacity(self.n_points);

        for i in (0..State::CAM_PARAM_SIZE * self.n_cams).step_by(State::CAM_PARAM_SIZE) {
            let u = self.camera_positions.fixed_rows::<3>(i);
            let w = self.camera_positions.fixed_rows::<3>(i + 3);
            let se3 = exp_se3(&u, &w);
            let rotation = Rotation3::<Float>::from_matrix(&se3.fixed_slice::<3,3>(0,0).into_owned());
            cam_positions.push(Isometry3::<Float>::new(se3.fixed_slice::<3,1>(0,3).into_owned(),rotation.scaled_axis()));
        }

        for i in 0..self.landmarks.len() {
            let point = self.landmarks[i];
            points.push(point.get_state().coords);
        }

        (cam_positions, points)
    }

    pub fn to_serial(&self) -> (Vec<[Float; 6]>, Vec<[Float; 3]>) {
        let mut cam_serial = Vec::<[Float; 6]>::with_capacity(self.n_cams);
        let mut points_serial = Vec::<[Float; 3]>::with_capacity(self.n_points);
        let number_of_cam_params = State::CAM_PARAM_SIZE * self.n_cams;

        for i in (0..number_of_cam_params).step_by(State::CAM_PARAM_SIZE) {
            let arr: [Float; State::CAM_PARAM_SIZE] = [
                self.camera_positions[i],
                self.camera_positions[i + 1],
                self.camera_positions[i + 2],
                self.camera_positions[i + 3],
                self.camera_positions[i + 4],
                self.camera_positions[i + 5],
            ];
            cam_serial.push(arr);
        }

        for i in 0..self.landmarks.len() {
            let vector = self.landmarks[i].get_state().coords;
            let arr: [Float; State::LANDMARK_PARAM_SIZE] = [vector[0], vector[1], vector[2]];
            points_serial.push(arr);
        }

        (cam_serial, points_serial)
    }

    pub fn from_serial((cam_serial, points_serial): &(Vec<[Float; State::CAM_PARAM_SIZE]>, Vec<[Float; State::LANDMARK_PARAM_SIZE]>)) -> State {
        let mut camera_positions = DVector::<Float>::zeros(State::CAM_PARAM_SIZE*cam_serial.len());
        let mut landmarks = Vec::<EuclideanLandmark>::with_capacity(points_serial.len());

        for i in 0..cam_serial.len() {
            let arr = cam_serial[i];
            let offset = 6 * i;
            camera_positions[offset] = arr[0];
            camera_positions[offset + 1] = arr[1];
            camera_positions[offset + 2] = arr[2];
            camera_positions[offset + 3] = arr[3];
            camera_positions[offset + 4] = arr[4];
            camera_positions[offset + 5] = arr[5];
        }

        for i in 0..points_serial.len() {
            let arr = points_serial[i];
            landmarks.push(EuclideanLandmark::new(arr[0], arr[1], arr[2]));
        }

        State::new(camera_positions,landmarks , cam_serial.len(), points_serial.len())
    }
}
