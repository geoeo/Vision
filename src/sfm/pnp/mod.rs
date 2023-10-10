extern crate nalgebra as na;

use crate::image::features::solver_feature::SolverFeature;
use crate::image::features::Feature;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sensors::camera::Camera;
use crate::sfm::{
    state::{CAMERA_PARAM_SIZE,pnp_state_linearizer::{get_euclidean_landmark_state,get_observed_features}, State},
    landmark::euclidean_landmark::EuclideanLandmark, pnp::pnp_config::PnPConfig,
};
use crate::Float;
use na::{base::Scalar, RealField};
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
pub mod pnp_config;


pub fn run_pnp<
    'a,
    F: serde::Serialize + float::Float + Scalar + RealField + SupersetOf<Float>,
    C: Camera<Float> + Copy + Send + Sync +'a + 'static,
    Feat: Feature + Clone + PartialEq + Eq + Hash + SolverFeature
>(
    pnp_config: &'a PnPConfig<C, Feat>,
    runtime_parameters: &'a RuntimeParameters<F>,
) -> 
    (  State<F, EuclideanLandmark<F>, 3>,
        Option<Vec<(Vec<[F; 6]>, Vec<[F; 3]>)>>
) {
    let mut state = get_euclidean_landmark_state::<F,Feat>(pnp_config.get_landmarks(), pnp_config.get_camera_pose_option());
    let observed_features = get_observed_features(pnp_config.get_features_norm());
    let cameras = vec![pnp_config.get_camera_norm()];


    let (tx_result, rx_result) = mpsc::channel::<(State<F, EuclideanLandmark<F>, 3>,Option<Vec<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; 3]>)>>)>();
    let (tx_abort, rx_abort) = mpsc::channel::<bool>();
    let (tx_done, rx_done) = mpsc::channel::<bool>();

    thread::scope(|s| {   
        s.spawn(move || {
            let solver = solver::Solver::<F, C, _, 3>::new();
            let some_debug_state_list = solver.solve(
                &mut state,
                &cameras,
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

    rx_result
    .recv()
    .expect("Did not receive state from solver thread!")

}
