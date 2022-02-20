extern crate nalgebra as na;

use na::{Matrix3, Vector3, Isometry3};
use crate::image::{epipolar,features::{Match,Feature}};
use crate::sensors::camera::Camera;
use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::sfm::{landmark::Landmark,bundle_adjustment::{camera_feature_map::CameraFeatureMap}};
use crate::Float;


pub mod camera_feature_map;
pub mod solver;
pub mod state;



pub fn run_ba<C : Camera + Copy, T : Feature>(all_matches: &Vec<Vec<Match<T>>>, camera_data: &Vec<((usize, C),(usize,C))>, 
                                img_dim : (usize,usize) ,runtime_parameters: &RuntimeParameters, pyramid_scale: Float, use_esstial_decomp_for_initial_guess: bool ) 
                                -> ((Vec<Isometry3<Float>>, Vec<Vector3<Float>>), (serde_yaml::Result<String>, serde_yaml::Result<String>)){
    let mut unique_cameras_sorted = Vec::<(usize, C)>::with_capacity(camera_data.len());
    for (a,b) in camera_data {
        unique_cameras_sorted.push(*a);
        unique_cameras_sorted.push(*b);
    }

    unique_cameras_sorted.sort_unstable_by(|(v1,_),(v2,_)| v1.partial_cmp(v2).unwrap());
    unique_cameras_sorted.dedup_by(|(v1,_),(v2,_)| v1==v2);

    let unique_camera_ids_sorted = unique_cameras_sorted.iter().map(|(id,_)| *id as u64).collect();
    let unique_camera_id_pairs = camera_data.iter().map(|((v1,_),(v2,_))| (*v1 as u64,*v2 as u64)).collect();
    let unique_cameras_sorted_by_id = unique_cameras_sorted.iter().map(|(_,cam)| *cam).collect::<Vec<C>>();

    let mut feature_map = CameraFeatureMap::new(all_matches,unique_camera_ids_sorted, img_dim);
    feature_map.add_matches(&unique_camera_id_pairs,all_matches, pyramid_scale);

    let initial_cam_pos_guess: Option<&Vec<(Vector3<Float>,Matrix3<Float>)>> = match use_esstial_decomp_for_initial_guess {
        false => None,
        true => {
            // let feature_machtes = matches.iter().map(|m| epipolar::extract_matches(m, orb_params_1_2.pyramid_scale, false)).collect::<Vec<Vec<(Vector2<Float>,Vector2<Float>)>>>();
            // let fundamental_matrices = feature_machtes.iter().map(|m| epipolar::eight_point(m)).collect::<Vec<epipolar::Fundamental>>();
            // let essential_matrices = fundamental_matrices.iter().enumerate().map(|(i,f)| {
            //     let (id_1,id_2) = image_id_pairs[i];
            //     let (c_1, _ ) = feature_map.camera_map[&id_1];
            //     let (c_2, _ ) = feature_map.camera_map[&id_2];
            //     epipolar::compute_essential(f, &cameras[c_1].get_projection(), &cameras[c_2].get_projection())
            // }).collect::<Vec<epipolar::Essential>>();
        
            // let normalized_matches = fundamental_matrices.iter().zip(feature_machtes.iter()).map(|(f,m)| epipolar::filter_matches(f, m)).collect::<Vec<Vec<(Vector3<Float>,Vector3<Float>)>>>();
            // //let initial_motion_decomp = essential_matrices.iter().enumerate().map(|(i,e)| epipolar::decompose_essential_förstner(e,&normalized_matches[i])).collect::<Vec<(Vector3<Float>,Matrix3<Float>)>>();
            // let initial_motion_decomp = essential_matrices.iter().enumerate().map(|(i,e)| epipolar::decompose_essential_kanatani(e,&normalized_matches[i])).collect::<Vec<(Vector3<Float>,Matrix3<Float>)>>();
            None
        
        } //TODO
    };

    //TODO: switch impl
    let mut state = feature_map.get_euclidean_landmark_state(None, Vector3::<Float>::new(0.0,0.0,-1.0));
    //let mut state = feature_map.get_inverse_depth_landmark_state(Some(&initial_motion_decomp), 1.0,&cameras);

    let observed_features = feature_map.get_observed_features(false);

    
    let some_debug_state_list = solver::optimize(&mut state, &unique_cameras_sorted_by_id, &observed_features, &runtime_parameters);
    let state_serialized = serde_yaml::to_string(&state.to_serial());
    let debug_states_serialized = serde_yaml::to_string(&some_debug_state_list);

    
    (state.as_matrix_point(), (state_serialized,debug_states_serialized))


}
