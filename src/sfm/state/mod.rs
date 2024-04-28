extern crate nalgebra as na;
extern crate simba;

use std::collections::HashMap;
use na::{DVector, SMatrix, Matrix4, Vector3, Vector6, Isometry3};
use crate::sfm::landmark::{Landmark,euclidean_landmark::EuclideanLandmark};
use crate::GenericFloat;
use cam_extrinsic_state::{CAMERA_PARAM_SIZE, CameraExtrinsicState};

pub mod ba_state_linearizer;
pub mod pnp_state_linearizer;
pub mod cam_extrinsic_state;

/**
 * This is ordered [cam_1,cam_2,..,cam_n,point_1,point_2,...,point_m]
 * cam is parameterized by [u_1,u_2,u_3,w_1,w_2,w_3]
 * point is parameterized by [x,y,z]
 * */
pub struct State<F: GenericFloat, L: Landmark<F,T>, const T: usize> {
    camera_parameters: Vec<CameraExtrinsicState<F>>, 
    landmarks: Vec<L>,
    pub camera_id_map: HashMap<usize, usize>, //Map of cam id to index in cam positions
    pub camera_id_by_idx: Vec<usize>,
    pub n_cams: usize,
    pub n_points: usize,
}

impl<F: GenericFloat, L: Landmark<F,T>, const T: usize> Clone for State<F,L,T> {
    fn clone(&self) -> State<F,L,T> {
        State::<F,L,T> {
            camera_parameters: self.camera_parameters.clone(),
            landmarks: self.landmarks.iter().map(|l| l.duplicate()).collect(), 
            camera_id_map: self.camera_id_map.clone(),
            camera_id_by_idx: self.camera_id_by_idx.clone(),
            n_cams: self.n_cams, 
            n_points: self.n_points
        }
    }
}

impl<F: GenericFloat, L: Landmark<F,T>, const T: usize> State<F,L,T> {
    pub fn new(raw_camera_parameters: DVector<F>, landmarks:  Vec<L>, camera_id_map: &HashMap<usize, usize>, n_cams: usize, n_points: usize) -> State<F,L,T> {
        let mut camera_parameters = Vec::<CameraExtrinsicState<F>>::with_capacity(n_cams);
        let camera_id_by_idx = Self::generate_camera_id_by_idx_vec(camera_id_map);

        for i in 0..n_cams {
            let offset = CAMERA_PARAM_SIZE * i;
            let arr = raw_camera_parameters.fixed_rows::<CAMERA_PARAM_SIZE>(offset);
            camera_parameters.push(CameraExtrinsicState::new(arr.into_owned()));
        }
        State{camera_parameters, landmarks, camera_id_map: camera_id_map.clone(),camera_id_by_idx , n_cams, n_points}
    }

    fn generate_camera_id_by_idx_vec(camera_id_map: &HashMap<usize, usize>) -> Vec<usize> {
        let mut cam_map_kvs = camera_id_map.iter().collect::<Vec<(_,_)>>();
        // Sort by index
        cam_map_kvs.sort_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap());
        cam_map_kvs.into_iter().map(|(k,_)| *k).collect::<Vec<_>>()
    }

    pub fn get_landmarks(&self) -> &Vec<L> {
        &self.landmarks
    }

    pub fn get_camera_positions(&self) -> Vec<Isometry3<F>> {
        self.camera_parameters.iter().map(|s| s.get_position()).collect()
    }

    pub fn get_camera_id_map(&self) -> &HashMap<usize, usize> {
        &self.camera_id_map
    }

    pub fn copy_from(&mut self, other: &State<F,L,T>) -> () {
        assert!(self.n_cams == other.n_cams);
        assert!(self.n_points == other.n_points);
        self.camera_parameters.copy_from_slice(&other.camera_parameters);
        self.landmarks = other.landmarks.iter().map(|l| l.duplicate()).collect()
    }

    pub fn update(&mut self, perturb: &DVector<F>) -> () {
        for i in 0..self.camera_parameters.len() {
            let linear_idx = i*CAMERA_PARAM_SIZE;   
            self.camera_parameters[i].update(perturb.fixed_rows::<CAMERA_PARAM_SIZE>(linear_idx).into_owned())     
        }
        
        let landmark_offset = self.camera_parameters.len()*CAMERA_PARAM_SIZE;
        for i in (landmark_offset..perturb.len()).step_by(T) {
            let landmark_id = (i-landmark_offset)/T;
            self.landmarks[landmark_id].update(&perturb.fixed_view::<T,1>(i,0).into());
        }
    }

    pub fn jacobian_wrt_world_coordiantes(&self, point_index: usize, cam_idx: usize) ->  SMatrix<F,3,T> {
        let state_idx = cam_idx/CAMERA_PARAM_SIZE;
        let iso = self.camera_parameters[state_idx].get_position();
        self.landmarks[point_index].jacobian(&iso)
    }

    /**
     * cam_idx is the index of a solver i.e. in CAMERA_PARAM_SIZE space
     */
    pub fn to_se3(&self, cam_idx: usize) -> Matrix4<F> {
        assert!(cam_idx < self.n_cams*CAMERA_PARAM_SIZE);
        let state_idx = cam_idx/CAMERA_PARAM_SIZE;
        self.camera_parameters[state_idx].get_position().to_matrix()
    }

    pub fn as_matrix_point(&self) -> (Vec<Isometry3<F>>, Vec<Vector3<F>>) {
        (self.get_camera_positions(), 
        self.landmarks.iter().map(|l| l.get_euclidean_representation().coords).collect::<Vec<_>>())
    }

    pub fn to_serial(&self) -> (Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; T]>) {
        let mut cam_serial = Vec::<[F; CAMERA_PARAM_SIZE]>::with_capacity(self.n_cams);
        let mut points_serial = Vec::<[F; T]>::with_capacity(self.n_points);

        for i in 0..self.n_cams {
            cam_serial.push(self.camera_parameters[i].to_serial());
        }

        for i in 0..self.landmarks.len() {
            points_serial.push(self.landmarks[i].get_state_as_array());
        }

        (cam_serial, points_serial)
    }

    pub fn from_serial((cam_serial, points_serial): &(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; T]>)) -> State<F,L,T> {
        let mut camera_parameters = Vec::<CameraExtrinsicState<F>>::with_capacity(cam_serial.len());
        let mut landmarks = Vec::<L>::with_capacity(points_serial.len());
        let mut camera_id_map = HashMap::<usize,usize>::with_capacity(camera_parameters.len());

        for i in 0..cam_serial.len() {
            let arr = cam_serial[i];
            let raw_state = Vector6::<F>::new(arr[0], arr[1], arr[2],arr[3],arr[4],arr[5]);
            camera_parameters.push(CameraExtrinsicState::<F>::new(raw_state));

            //TODO:Check this map
            camera_id_map.insert(i,i);
        }

        for i in 0..points_serial.len() {
            landmarks.push(L::from_array(&points_serial[i]));
        }

        let camera_id_by_idx = Self::generate_camera_id_by_idx_vec(&camera_id_map);
        State {
            camera_parameters,
            landmarks,
            camera_id_map,
            camera_id_by_idx,
            n_cams: cam_serial.len(),
            n_points: points_serial.len()
        }
    }

    pub fn to_euclidean_landmarks(&self) -> State<F, EuclideanLandmark<F>, 3> {
        State::<F,EuclideanLandmark<F>,3> {
            camera_parameters: self.camera_parameters.clone(),
            landmarks:  self.get_landmarks().iter().map(|l| EuclideanLandmark::<F>::from_state_with_id(l.get_euclidean_representation().coords,&l.get_id())).collect(), 
            camera_id_map: self.camera_id_map.clone(),
            camera_id_by_idx: self.camera_id_by_idx.clone(),
            n_cams: self.n_cams, 
            n_points: self.n_points
        }
    }
}
