extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use crate::image::features::solver_feature::SolverFeature;
use crate::image::features::Feature;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sensors::camera::Camera;
use crate::sfm::{
    state::{cam_state::CamState,
    pnp_state_linearizer::{get_euclidean_landmark_state,get_observed_features},
    State,landmark::{euclidean_landmark::LANDMARK_PARAM_SIZE,euclidean_landmark::EuclideanLandmark}},
    pnp::pnp_config::PnPConfig,
};
use crate::{GenericFloat,Float};
use std::{
    marker::{Send, Sync},
    sync::mpsc,
    thread,
};
use termion::input::TermRead;

pub mod solver;
pub mod pnp_config;

pub fn run_pnp<
    'a,
    F: serde::Serialize + GenericFloat,
    const CP: usize,
    CS: CamState<F,C,CP> + Copy + Send + Sync + 'static,
    CConfig: Camera<Float> + Clone + Copy + Send + Sync +'a + 'static,
    C: Camera<F> + Clone + Copy + Send + Sync +'a + 'static,
    Feat: Feature + SolverFeature
>(
    pnp_config: &'a PnPConfig<CConfig, Feat>,
    runtime_parameters: &'a RuntimeParameters<F>,
) -> 
    (  State<F, C, EuclideanLandmark<F>,CS, LANDMARK_PARAM_SIZE, CP>,
        Option<Vec<(Vec<[F; CP]>, Vec<[F; LANDMARK_PARAM_SIZE]>)>>
) {
    let mut state = get_euclidean_landmark_state::<F,Feat,CConfig,C,CS,CP>(pnp_config.get_landmarks(), pnp_config.get_camera_pose_option(), pnp_config.get_camera_norm());
    let observed_features = get_observed_features(pnp_config.get_features_norm());

    let (tx_result, rx_result) = mpsc::channel::<(State<F, C,EuclideanLandmark<F>,CS, LANDMARK_PARAM_SIZE, CP>,Option<Vec<(Vec<[F; CP]>, Vec<[F; LANDMARK_PARAM_SIZE]>)>>)>();
    let (tx_abort, rx_abort) = mpsc::channel::<bool>();
    let (tx_done, rx_done) = mpsc::channel::<bool>();

    thread::scope(|s| {   
        s.spawn(move || {
            let solver = solver::Solver::<F, C, _,_, LANDMARK_PARAM_SIZE, CP>::new();
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

            tx_result
                .send((state,debug_state))
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
