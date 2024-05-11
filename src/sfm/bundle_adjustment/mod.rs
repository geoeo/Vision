extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use crate::image::features::solver_feature::SolverFeature;
use crate::image::features::Feature;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sensors::camera::Camera;
use crate::sfm::{
    bundle_adjustment::ba_config::{BAConfig,conversions::generate_abs_landmark_map},
    state::{
        State,
        ba_state_linearizer::BAStateLinearizer, 
        cam_state::CamState,
        //cam_state::cam_extrinsic_state::{CAMERA_PARAM_SIZE, CameraExtrinsicState},
        landmark::{Landmark,euclidean_landmark::EuclideanLandmark,euclidean_landmark, inverse_depth_landmark}}, 
};
use crate::{GenericFloat,Float};
use std::{
    marker::{Send, Sync},
    sync::mpsc,
    thread,
    collections::HashSet
};
use termion::input::TermRead;

pub mod solver;
pub mod ba_config;

pub fn run_ba<
    F: serde::Serialize + GenericFloat,
    const LP: usize,
    const CP: usize,
    CS: CamState<F,C,CP> + Copy + Send + Sync + 'static,
    L: Landmark<F,LP>,
    CConfig: Camera<Float> + Copy + Send + Sync + 'static,
    C: Camera<F> + Copy + Send + Sync + 'static,
    Feat: Feature + SolverFeature + 'static
>(
    sfm_config: &BAConfig<CConfig, Feat>,
    runtime_parameters: &RuntimeParameters<F>,
    trajectories: &Vec<Vec<(usize,usize)>>
) -> (
    State<F, C,EuclideanLandmark<F>,CS, {euclidean_landmark::LANDMARK_PARAM_SIZE}, CP>,
    Option<Vec<(Vec<[F; CP]>, Vec<[F; euclidean_landmark::LANDMARK_PARAM_SIZE]>)>>
) {
    let abs_landmark_map = generate_abs_landmark_map(sfm_config.root(),sfm_config.paths(),sfm_config.landmark_map(),sfm_config.abs_pose_map());
    let paths = trajectories.clone().into_iter().flatten().collect::<Vec<(usize,usize)>>();

    // We only consider a potential subset of the paths for the solver
    let unique_landmark_id_set = paths.iter().map(|p| abs_landmark_map.get(p).expect("No landmarks for path")).flatten().map(|l| l.get_id().expect("No id")).collect::<HashSet<_>>();
    let state_linearizer = BAStateLinearizer::new(&paths,&unique_landmark_id_set); // This works

    let (tx_result, rx_result) = mpsc::channel::<(State<F,C, EuclideanLandmark<F>,CS, {euclidean_landmark::LANDMARK_PARAM_SIZE},CP>, Option<Vec<(Vec<[F; CP]>, Vec<[F; 3]>)>>)>();
    let (tx_abort, rx_abort) = mpsc::channel::<bool>();
    let (tx_done, rx_done) = mpsc::channel::<bool>();

    thread::scope(|s| {   
        s.spawn(move || {

            let (s,d) = match LP {
                euclidean_landmark::LANDMARK_PARAM_SIZE => {
                    let solver = solver::Solver::<F, C, _, _, {euclidean_landmark::LANDMARK_PARAM_SIZE}, CP>::new();
                    let (mut state, observed_features) = state_linearizer.get_euclidean_landmark_state(
                        &paths,
                        sfm_config.match_norm_map(),
                        sfm_config.abs_pose_map(),
                        &abs_landmark_map,
                        sfm_config.reprojection_error_map(),
                        sfm_config.camera_norm_map()
                    );
                    let some_debug_state_list = solver.solve(
                        &mut state,
                        &observed_features, 
                        runtime_parameters,
                        Some(&rx_abort),
                        Some(&tx_done),
                    );
                    let debug_state = match some_debug_state_list {
                        None => None,
                        Some(list) => Some(list.iter().map(|s| s.to_euclidean_landmarks().to_serial()).collect())
                    };

                    let s = state.to_euclidean_landmarks();
                    (s,debug_state)
                },
                inverse_depth_landmark::LANDMARK_PARAM_SIZE => {
                    let solver = solver::Solver::<F, C, _, _, {inverse_depth_landmark::LANDMARK_PARAM_SIZE}, CP>::new();
                    let (mut state, observed_features) = state_linearizer.get_inverse_depth_landmark_state(
                        &paths,
                        sfm_config.match_norm_map(),
                        sfm_config.abs_pose_map(),
                        sfm_config.reprojection_error_map(),
                        sfm_config.camera_norm_map()
                    );
                    let some_debug_state_list = solver.solve(
                        &mut state,
                        &observed_features,
                        runtime_parameters,
                        Some(&rx_abort),
                        Some(&tx_done),
                    );
                    let debug_state = match some_debug_state_list {
                        None => None,
                        Some(list) => Some(list.iter().map(|s| s.to_euclidean_landmarks().to_serial()).collect())
                    };

                    let s = state.to_euclidean_landmarks();
                    (s,debug_state)
                },
                _ => panic!("Invalid Landmark Param")
        
            };

            tx_result
                .send((s,d))
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

    rx_result
        .recv()
        .expect("Did not receive state from solver thread!")

}
