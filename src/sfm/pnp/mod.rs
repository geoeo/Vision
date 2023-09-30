extern crate nalgebra as na;

use crate::image::features::solver_feature::SolverFeature;
use crate::image::features::Feature;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sensors::camera::Camera;
use crate::sfm::{
    state,
    state::state_linearizer::StateLinearizer,
    landmark::euclidean_landmark::EuclideanLandmark, pnp::pnp_config::PnPConfig,
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
pub mod pnp_config;


pub fn run_pnp<
    'a,
    F: serde::Serialize + float::Float + Scalar + RealField + SupersetOf<Float>,
    C: Camera<Float> + Copy + Send + Sync +'a + 'static,
    T: Feature + Clone + PartialEq + Eq + Hash + SolverFeature
>(
    sfm_config: &'a PnPConfig<C, T>,
    runtime_parameters: &'a RuntimeParameters<F>,
) -> (
    (Vec<Isometry3<F>>, Vec<Vector3<F>>),
    (serde_yaml::Result<String>, serde_yaml::Result<String>),
) {
    panic!("TODO");
}
