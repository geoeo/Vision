extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use na::{Matrix3, Vector3, Isometry3, SimdRealField, ComplexField,base::Scalar, RealField};
use simba::scalar::{SubsetOf,SupersetOf};
use std::{ops::Mul,convert::From};
use num_traits::{float,NumAssign};
use crate::image::features::solver_feature::SolverFeature;
use crate::image::features::{Match,Feature};
use crate::sensors::camera::Camera;
use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::sfm::{SFMConfig,landmark::Landmark,bundle_adjustment::{camera_feature_map::CameraFeatureMap}};
use crate::Float;


pub mod camera_feature_map;
pub mod solver;
pub mod state;

pub fn run_ba<F: serde::Serialize + float::Float + Scalar + NumAssign + SimdRealField + ComplexField + Mul<F> + From<F> + RealField + SubsetOf<Float> + SupersetOf<Float>, C : Camera<Float> + Copy, C2: Camera<F> + Copy, T : Feature + Clone + PartialEq + SolverFeature>(
    matches: &Vec<Vec<Vec<Match<T>>>>, sfm_config: &SFMConfig<C, C2, T>,img_dim : (usize,usize) ,runtime_parameters: &RuntimeParameters<F>) 
                                -> ((Vec<Isometry3<F>>, Vec<Vector3<F>>), (serde_yaml::Result<String>, serde_yaml::Result<String>)){


    let (unique_camera_ids_sorted,_) = sfm_config.compute_unqiue_ids_cameras_root_first();
    let (_,unique_cameras_sorted_by_id_ba) = sfm_config.compute_unqiue_ids_cameras_ba_root_first();
    let path_id_pairs = sfm_config.compute_path_id_pairs();

    let mut feature_map = CameraFeatureMap::new(sfm_config.match_map(),unique_camera_ids_sorted, img_dim);
    feature_map.add_matches(sfm_config.match_map());

    //TODO: switch impl on landmark state
    let mut state = feature_map.get_euclidean_landmark_state(
        &path_id_pairs, 
        sfm_config.match_map(), 
        sfm_config.pose_map(), 
        sfm_config.landmark_map(), 
        sfm_config.reprojection_error_map(),  
    );
    //let mut state = feature_map.get_inverse_depth_landmark_state(Some(&initial_motion_decomp), depth_prior,&cameras);
    
    //TODO: check this
    let observed_features = feature_map.get_observed_features::<F>(false);
    
    let some_debug_state_list = solver::optimize::<_,_,_,3>(&mut state, &unique_cameras_sorted_by_id_ba, &observed_features, runtime_parameters);
    let state_serialized = serde_yaml::to_string(&state.to_serial());
    let debug_states_serialized = serde_yaml::to_string(&some_debug_state_list);

    
    (state.as_matrix_point(), (state_serialized,debug_states_serialized))


}
