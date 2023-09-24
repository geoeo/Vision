extern crate nalgebra as na;
extern crate num_traits;


use std::marker::{Send,Sync};
use na::{DVector,DMatrix,Point3, Vector4, base::Scalar, RealField, convert};
use simba::scalar::SupersetOf;
use num_traits::float;

use crate::sensors::camera::Camera;
use crate::numerics::lie::left_jacobian_around_identity;
use crate::sfm::{landmark::Landmark,bundle_adjustment::{state::State, state_linearizer}, optimizer::Optimizer};
use crate::sfm::runtime_parameters::RuntimeParameters; 
use crate::Float;

const CAMERA_PARAM_SIZE: usize = 6; //TODO make this generic with state


struct Solver<F: SupersetOf<Float>, C : Camera<Float>, L: Landmark<F, LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync, const LANDMARK_PARAM_SIZE: usize>  where F: float::Float + Scalar + RealField {
    optimizer: Optimizer<F,C,L, LANDMARK_PARAM_SIZE>
}

impl<F: SupersetOf<Float>, C : Camera<Float>, L: Landmark<F, LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync, const LANDMARK_PARAM_SIZE: usize>  Solver<F,C,L,LANDMARK_PARAM_SIZE> where F: float::Float + Scalar + RealField  {

 fn get_feature_index_in_residual(&self, cam_id: usize, feature_id: usize, n_cams: usize) -> usize {
        2*(cam_id + feature_id*n_cams)
    }
    
 fn compute_residual(&self,estimated_features: &DVector<F>, observed_features: &DVector<F>, residual_vector: &mut DVector<F>) 
        -> () {
        assert_eq!(residual_vector.nrows(), estimated_features.nrows());
        for i in 0..residual_vector.nrows() {
            if observed_features[i] != convert(state_linearizer::NO_FEATURE_FLAG) {
                residual_vector[i] =  estimated_features[i] - observed_features[i];
            } else {
                residual_vector[i] = F::zero();
            }
            assert!(!residual_vector[i].is_nan());
        }
    }
    
    
    fn compute_jacobian_wrt_camera_extrinsics(&self, camera: &C, state: &State<F,L,LANDMARK_PARAM_SIZE>, cam_idx: usize, point: &Point3<F> ,i: usize, j: usize, jacobian: &mut DMatrix<F>) 
        -> () {
        let transformation = state.to_se3(cam_idx);
        let transformed_point = transformation*Vector4::<F>::new(point[0],point[1],point[2],F::one());
        let lie_jacobian = left_jacobian_around_identity(&transformed_point.fixed_rows::<3>(0)); 
    
        let projection_jacobian = camera.get_jacobian_with_respect_to_position_in_camera_frame(&transformed_point.fixed_rows::<3>(0)).expect("get_jacobian_with_respect_to_position_in_camera_frame failed!");
        let local_jacobian = projection_jacobian*lie_jacobian;
    
        jacobian.fixed_view_mut::<2,6>(i,j).copy_from(&local_jacobian);
    }
    
     fn compute_jacobian(&self, state: &State<F,L,LANDMARK_PARAM_SIZE>, cameras: &Vec<&C>, jacobian: &mut DMatrix<F>) 
        -> () {
        //cam
        assert_eq!(state.n_cams,1);
        let cam_state_idx = 0;
        let cam_id = cam_state_idx/CAMERA_PARAM_SIZE;
        let camera = cameras[cam_id];
        
        //landmark
        for point_id in 0..state.n_points {
            let point = state.get_landmarks()[point_id].get_euclidean_representation();
    
            let row = self.get_feature_index_in_residual(cam_id, point_id, state.n_cams);
            let a_j = cam_state_idx;
            
            self.compute_jacobian_wrt_camera_extrinsics(camera , state, cam_state_idx,&point,row,a_j, jacobian);
        }
    
        
    
    }

}

