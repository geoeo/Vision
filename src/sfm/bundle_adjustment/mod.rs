extern crate nalgebra as na;

use crate::image::features::solver_feature::SolverFeature;
use crate::image::features::Feature;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sensors::camera::Camera;
use crate::sfm::{
    landmark::euclidean_landmark::EuclideanLandmark, bundle_adjustment::ba_config::{BAConfig,compute_path_id_pairs},
    state::{State,ba_state_linearizer::BAStateLinearizer, CAMERA_PARAM_SIZE}, 
};
use crate::Float;
use na::{base::Scalar, Isometry3, RealField, Vector3};
use num_traits::float;
use simba::scalar::SupersetOf;
use std::{
    hash::Hash,
    marker::{Send, Sync},
    sync::mpsc,
    thread,
};
use termion::input::TermRead;

pub mod solver;
pub mod ba_config;

pub fn run_ba<
    'a,
    F: serde::Serialize + float::Float + Scalar + RealField + SupersetOf<Float>,
    C: Camera<Float> + Copy + Send + Sync +'a + 'static,
    Feat: Feature + Clone + PartialEq + Eq + Hash + SolverFeature
>(
    sfm_config: &'a BAConfig<C, Feat>,
    runtime_parameters: &'a RuntimeParameters<F>,
) -> (
    (Vec<Isometry3<F>>, Vec<Vector3<F>>),
    (serde_yaml::Result<String>, serde_yaml::Result<String>),
) {
    let (unique_camera_ids_sorted, unique_cameras_sorted_by_id) =
        sfm_config.compute_unqiue_ids_cameras_root_first();
    let path_id_pairs = compute_path_id_pairs(sfm_config.root(), sfm_config.paths());

    let state_linearizer = BAStateLinearizer::new(unique_camera_ids_sorted);

    //TODO: switch impl on landmark state

    let (mut state, feature_location_lookup) = state_linearizer.get_euclidean_landmark_state(
        &path_id_pairs,
        sfm_config.match_norm_map(),
        sfm_config.abs_pose_map(),
        sfm_config.abs_landmark_map(),
        sfm_config.reprojection_error_map(),
        sfm_config.unique_landmark_ids().len(),
    );
    let observed_features = state_linearizer.get_observed_features::<F>(
        &feature_location_lookup,
        sfm_config.unique_landmark_ids().len(),
    );

    let (tx_result, rx_result) = mpsc::channel::<(State<F, EuclideanLandmark<F>, 3>,Option<Vec<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; 3]>)>>)>();
    let (tx_abort, rx_abort) = mpsc::channel::<bool>();
    let (tx_done, rx_done) = mpsc::channel::<bool>();

    thread::scope(|s| {   
        s.spawn(move || {
            let solver = solver::Solver::<F, C, _, 3>::new();
            let some_debug_state_list = solver.solve(
                &mut state,
                &unique_cameras_sorted_by_id,
                &observed_features,
                runtime_parameters,
                Some(&rx_abort),
                Some(&tx_done),
            );
            tx_result
                .send((state,some_debug_state_list))
                .expect("Tx can not send state from solver thread");
        });

        s.spawn(move || {
            // Use asynchronous stdin
            let mut stdin = termion::async_stdin().keys();
            let mut solver_block = true;
            while solver_block {
                let input = stdin.next();

                if let Some(Ok(key)) = input {
                    let _ = match key {
                        termion::event::Key::Char('q') => tx_abort.send(true),
                        _ => Ok(()),
                    };
                }
                solver_block &= match rx_done.try_recv() {
                    Ok(b) => !b,
                    _ => true,
                };
            }
        });
    });

    let (state,some_debug_state_list) = rx_result
        .recv()
        .expect("Did not receive state from solver thread!");
    let state_serialized = serde_yaml::to_string(&state.to_serial());
    let debug_states_serialized = serde_yaml::to_string(&some_debug_state_list);

    (
        state.as_matrix_point(),
        (state_serialized, debug_states_serialized),
    )
}
