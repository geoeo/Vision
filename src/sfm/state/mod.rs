extern crate nalgebra as na;
extern crate simba;

use std::collections::HashMap;
use std::marker::PhantomData;
use na::{DVector, Isometry3, SMatrix, SVector, Vector3, Point3};
use cam_state::CamState;
use landmark::{Landmark,euclidean_landmark::EuclideanLandmark};
use crate::{
    GenericFloat,
    sensors::camera::Camera,
    numerics::lie::left_jacobian_around_identity
};


pub mod ba_state_linearizer;
pub mod pnp_state_linearizer;
pub mod cam_state;
pub mod landmark;

/**
 * This is ordered [cam_1,cam_2,..,cam_n,point_1,point_2,...,point_m]
 * cam is parameterized by [u_1,u_2,u_3,w_1,w_2,w_3]
 * point is parameterized by [x,y,z]
 * */
pub struct State<F: GenericFloat, C: Camera<F>,  L: Landmark<F,LANDMARK_PARMA_SIZE>, CS: CamState<F, C, CAMERA_PARAM_SIZE>, const LANDMARK_PARMA_SIZE: usize, const CAMERA_PARAM_SIZE: usize> {
    camera_parameters: Vec<CS>, 
    landmarks: Vec<L>,
    pub camera_id_map: HashMap<usize, usize>, //Map of cam id to index in cam positions, TODO: To Vector
    pub n_cams: usize,
    pub n_points: usize,
    _phantom_f: PhantomData<F>,
    _panthom_c: PhantomData<C>
}

impl<F: GenericFloat, C: Camera<F>,  L: Landmark<F,LANDMARK_PARMA_SIZE>, CS: CamState<F,C, CAMERA_PARAM_SIZE>, const LANDMARK_PARMA_SIZE: usize, const CAMERA_PARAM_SIZE: usize> Clone for State<F,C,L,CS,LANDMARK_PARMA_SIZE, CAMERA_PARAM_SIZE> {
    fn clone(&self) -> State<F,C,L,CS,LANDMARK_PARMA_SIZE, CAMERA_PARAM_SIZE> {
        State::<F,C,L,CS,LANDMARK_PARMA_SIZE, CAMERA_PARAM_SIZE> {
            camera_parameters: self.camera_parameters.iter().map(|c| c.duplicate()).collect(),
            landmarks: self.landmarks.iter().map(|l| l.duplicate()).collect(), 
            camera_id_map: self.camera_id_map.clone(),
            n_cams: self.n_cams, 
            n_points: self.n_points,
            _phantom_f: Default::default(),
            _panthom_c: Default::default()
        }
    }
}

impl<F: GenericFloat, C: Camera<F> + Copy, L: Landmark<F,LANDMARK_PARMA_SIZE>, CS: CamState<F, C, CAMERA_PARAM_SIZE>, const LANDMARK_PARMA_SIZE: usize, const CAMERA_PARAM_SIZE: usize> State<F,C,L,CS,LANDMARK_PARMA_SIZE, CAMERA_PARAM_SIZE> {
    pub fn new(raw_camera_parameters: DVector<F>, cameras_with_id: &Vec<(C,usize)>, landmarks:  Vec<L>, n_cams: usize, n_points: usize) -> State<F,C,L,CS,LANDMARK_PARMA_SIZE, CAMERA_PARAM_SIZE> {
        let mut camera_parameters = Vec::<CS>::with_capacity(n_cams);
        let mut camera_id_map = HashMap::<usize,usize>::with_capacity(n_cams);
        for i in 0..n_cams {
            let offset = CAMERA_PARAM_SIZE * i;
            let arr = raw_camera_parameters.fixed_rows::<CAMERA_PARAM_SIZE>(offset);
            let (camera,id) = cameras_with_id[i];
            camera_parameters.push(CS::new(arr.into_owned(),&camera));
            camera_id_map.insert(id,i);
        }
        State{camera_parameters, landmarks, camera_id_map, n_cams, n_points,  _phantom_f: Default::default(),_panthom_c: Default::default()}
    }

    pub fn get_landmarks(&self) -> &Vec<L> {
        &self.landmarks
    }

    pub fn get_camera_positions(&self) -> Vec<Isometry3<F>> {
        self.camera_parameters.iter().map(|s| s.get_position()).collect()
    }

    pub fn get_camera(&self, idx: usize) -> C {
        assert!(idx < self.n_cams);
        self.camera_parameters[idx].get_camera()
    }

    pub fn get_camera_id_map(&self) -> &HashMap<usize, usize> {
        &self.camera_id_map
    }

    pub fn copy_from(&mut self, other: &State<F,C,L,CS,LANDMARK_PARMA_SIZE, CAMERA_PARAM_SIZE>) -> () {
        assert!(self.n_cams == other.n_cams);
        assert!(self.n_points == other.n_points);
        self.camera_parameters = other.camera_parameters.iter().map(|c| c.duplicate()).collect();
        self.landmarks = other.landmarks.iter().map(|l| l.duplicate()).collect()
    }

    pub fn update(&mut self, perturb: &DVector<F>) -> () {
        for i in 0..self.camera_parameters.len() {
            let linear_idx = i*CAMERA_PARAM_SIZE;   
            self.camera_parameters[i].update(perturb.fixed_rows::<CAMERA_PARAM_SIZE>(linear_idx).into_owned())     
        }
        
        let landmark_offset = self.camera_parameters.len()*CAMERA_PARAM_SIZE;
        for i in (landmark_offset..perturb.len()).step_by(LANDMARK_PARMA_SIZE) {
            let landmark_id = (i-landmark_offset)/LANDMARK_PARMA_SIZE;
            self.landmarks[landmark_id].update(&perturb.fixed_view::<LANDMARK_PARMA_SIZE,1>(i,0).into());
        }
    }

    pub fn jacobian_wrt_landmarks(&self, point_index: usize, cam_idx: usize) ->  SMatrix<F,3,LANDMARK_PARMA_SIZE> {
        let state_idx = cam_idx/CAMERA_PARAM_SIZE;
        let iso = self.camera_parameters[state_idx].get_position();
        self.landmarks[point_index].jacobian(&iso)
    }

    pub fn jacobian_wrt_camera(&self, point: &Point3<F>, cam_idx: usize) ->  SMatrix<F,2,CAMERA_PARAM_SIZE> {
        assert!(cam_idx < self.n_cams);
        let camera_parameters = &self.camera_parameters[cam_idx];
        let transformed_point = camera_parameters.get_position().transform_point(point);
        let lie_jacobian = left_jacobian_around_identity(&transformed_point.coords);
        camera_parameters.get_jacobian(&transformed_point, &lie_jacobian)
    }

    /**
     * cam_idx is the index of a solver i.e. in CAMERA_PARAM_SIZE space
     */
    pub fn to_isometry(&self, cam_idx: usize) -> Isometry3<F> {
        assert!(cam_idx < self.n_cams*CAMERA_PARAM_SIZE);
        let state_idx = cam_idx/CAMERA_PARAM_SIZE;
        self.camera_parameters[state_idx].get_position()
    }

    pub fn as_matrix_point(&self) -> (Vec<Isometry3<F>>, Vec<Vector3<F>>) {
        (self.get_camera_positions(), 
        self.landmarks.iter().map(|l| l.get_euclidean_representation().coords).collect::<Vec<_>>())
    }

    pub fn to_serial(&self) -> (Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; LANDMARK_PARMA_SIZE]>) {
        let mut cam_serial = Vec::<[F; CAMERA_PARAM_SIZE]>::with_capacity(self.n_cams);
        let mut points_serial = Vec::<[F; LANDMARK_PARMA_SIZE]>::with_capacity(self.n_points);

        for i in 0..self.n_cams {
            cam_serial.push(self.camera_parameters[i].to_serial());
        }

        for i in 0..self.landmarks.len() {
            points_serial.push(self.landmarks[i].get_state_as_array());
        }

        (cam_serial, points_serial)
    }

    pub fn from_serial((cam_serial, points_serial): &(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; LANDMARK_PARMA_SIZE]>), camera: &C) -> State<F,C,L,CS,LANDMARK_PARMA_SIZE, CAMERA_PARAM_SIZE> {
        let mut camera_parameters = Vec::<CS>::with_capacity(cam_serial.len());
        let mut landmarks = Vec::<L>::with_capacity(points_serial.len());
        let mut camera_id_map = HashMap::<usize,usize>::with_capacity(camera_parameters.len());

        for i in 0..cam_serial.len() {
            let arr = cam_serial[i];
            let mut raw_state = SVector::<F,CAMERA_PARAM_SIZE>::zeros();
            for j in 0..CAMERA_PARAM_SIZE {
                raw_state[j] = arr[j]
            }
            camera_parameters.push(CS::new(raw_state, camera));

            //TODO:Check this map
            camera_id_map.insert(i,i);
        }

        for i in 0..points_serial.len() {
            landmarks.push(L::from_array(&points_serial[i]));
        }

        State {
            camera_parameters,
            landmarks,
            camera_id_map,
            n_cams: cam_serial.len(),
            n_points: points_serial.len(),
            _phantom_f: Default::default(),
            _panthom_c: Default::default()
        }
    }

    pub fn to_euclidean_landmarks(&self) -> State<F, C, EuclideanLandmark<F>, CS, 3, CAMERA_PARAM_SIZE> {
        State::<F,C,EuclideanLandmark<F>,CS,3, CAMERA_PARAM_SIZE> {
            camera_parameters: self.camera_parameters.iter().map(|c| c.duplicate()).collect(),
            landmarks:  self.get_landmarks().iter().map(|l| EuclideanLandmark::<F>::from_state_with_id(l.get_euclidean_representation().coords,&l.get_id())).collect(), 
            camera_id_map: self.camera_id_map.clone(),
            n_cams: self.n_cams, 
            n_points: self.n_points,
            _phantom_f: Default::default(),
            _panthom_c: Default::default()
        }
    }
}
