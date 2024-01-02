extern crate nalgebra as na;
extern crate num_traits;

use crate::image::features::solver_feature::SolverFeature;
use crate::image::features::Feature;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sensors::camera::Camera;
use crate::sfm::{
    landmark::euclidean_landmark::EuclideanLandmark, bundle_adjustment::ba_config::{BAConfig,conversions::generate_abs_landmark_map},
    state::{State,ba_state_linearizer::BAStateLinearizer, CAMERA_PARAM_SIZE}, 
};
use crate::Float;
use na::{base::Scalar, RealField};
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
    F: serde::Serialize + Scalar + RealField + SupersetOf<Float> + Copy + num_traits::Float,
    C: Camera<Float> + Copy + Send + Sync +'a + 'static,
    Feat: Feature + Clone + PartialEq + Eq + Hash + SolverFeature
>(
    sfm_config: &'a BAConfig<C, Feat>,
    runtime_parameters: &'a RuntimeParameters<F>,
    trajectories: &Vec<Vec<(usize,usize)>>
) -> (
    State<F, EuclideanLandmark<F>, 3>,
    Option<Vec<(Vec<[F; 6]>, Vec<[F; 3]>)>>
) {
    
    let abs_landmark_map = generate_abs_landmark_map(sfm_config.root(),sfm_config.paths(),sfm_config.landmark_map(),sfm_config.abs_pose_map());
    let paths = trajectories.clone().into_iter().flatten().collect::<Vec<(usize,usize)>>();
    let state_linearizer = BAStateLinearizer::new(&paths,&abs_landmark_map);

    //TODO: switch impl on landmark state

    //TODO: number of unique landmarks is taken from the config. With a subset of trajectories this is no longer valid.
    //Consecuitve landmark indexing must be done here
    let (mut state, observed_features) = state_linearizer.get_euclidean_landmark_state(
        &paths,
        sfm_config.match_norm_map(),
        sfm_config.abs_pose_map(),
        &abs_landmark_map,
        sfm_config.reprojection_error_map()
    );


    let (tx_result, rx_result) = mpsc::channel::<(State<F, EuclideanLandmark<F>, 3>,Option<Vec<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; 3]>)>>)>();
    let (tx_abort, rx_abort) = mpsc::channel::<bool>();
    let (tx_done, rx_done) = mpsc::channel::<bool>();
    let camera_map = sfm_config.camera_norm_map();

    thread::scope(|s| {   
        s.spawn(move || {
            let solver = solver::Solver::<F, C, _, 3>::new();
            let some_debug_state_list = solver.solve(
                &mut state,
                &camera_map,
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
