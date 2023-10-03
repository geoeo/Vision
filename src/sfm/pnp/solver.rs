extern crate nalgebra as na;
extern crate num_traits;

use na::{base::Scalar, convert, DMatrix, DVector, Point3, RealField, Vector4, Matrix, Dyn, U4, VecStorage};
use num_traits::float;
use simba::scalar::SupersetOf;
use std::marker::{Send, Sync};
use std::sync::mpsc;

use crate::numerics::{lie::left_jacobian_around_identity, optimizer::gauss_newton::OptimizerGn};
use crate::sensors::camera::Camera;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sfm::{
    landmark::Landmark,
    state::{ba_state_linearizer, State},
};
use crate::Float;

const CAMERA_PARAM_SIZE: usize = 6; //TODO make this generic with state

pub struct Solver<
    F: SupersetOf<Float>,
    C: Camera<Float> + 'static,
    L: Landmark<F, LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync + 'static,
    const LANDMARK_PARAM_SIZE: usize,
> where
    F: float::Float + Scalar + RealField,
{
    optimizer: OptimizerGn<F, C, L, LANDMARK_PARAM_SIZE>,
}

impl<
        F: SupersetOf<Float>,
        C: Camera<Float> + 'static,
        L: Landmark<F, LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync + 'static,
        const LANDMARK_PARAM_SIZE: usize,
    > Solver<F, C, L, LANDMARK_PARAM_SIZE>
where
    F: float::Float + Scalar + RealField,
{
    pub fn new() -> Solver<F, C, L, LANDMARK_PARAM_SIZE> {
        Solver {
            optimizer: OptimizerGn::new(
                Box::new(Self::get_estimated_features),
                Box::new(Self::compute_residual),
                Box::new(Self::compute_jacobian),
                Box::new(Self::compute_state_size),
            ),
        }
    }

    /**
     * In the format [f1_cam1, f1_cam2,...]
     * Some entries may be 0 since not all cams see all points
     * */
    fn get_estimated_features(
        state: &State<F, L, LANDMARK_PARAM_SIZE>,
        cameras: &Vec<&C>,
        observed_features: &DVector<F>, //TODO: remove this
        estimated_features: &mut DVector<F>,
    ) -> () {
        let n_cams = state.n_cams;
        let n_points = state.n_points;
        assert_eq!(estimated_features.nrows(), 2 * n_points * n_cams);
        let mut position_world =
            Matrix::<F, U4, Dyn, VecStorage<F, U4, Dyn>>::from_element(n_points, F::one());
        for j in 0..n_points {
            position_world.fixed_view_mut::<3, 1>(0, j).copy_from(
                &state.get_landmarks()[j]
                    .get_euclidean_representation()
                    .coords,
            );
        }
        for i in 0..n_cams {
            let cam_idx = 6 * i;
            let pose = state.to_se3(cam_idx);
            let camera = cameras[i];

            //TODO: use transform_into_other_camera_frame
            let transformed_points = pose * &position_world;
            for j in 0..n_points {
                let estimated_feature =
                    camera.project(&transformed_points.fixed_view::<3, 1>(0, j));

                let feat_id = Self::get_feature_index_in_residual(i, j, n_cams);
                let est = estimated_feature.unwrap();
                estimated_features[feat_id] = est.x;
                estimated_features[feat_id + 1] = est.y;
                
            }
        }
    }

    //TODO: ncams is unneccessary
    fn get_feature_index_in_residual(cam_id: usize, feature_id: usize, n_cams: usize) -> usize {
        2 * feature_id
    }

    fn compute_residual(
        estimated_features: &DVector<F>,
        observed_features: &DVector<F>,
        residual_vector: &mut DVector<F>,
    ) -> () {
        assert_eq!(residual_vector.nrows(), estimated_features.nrows());
        for i in 0..residual_vector.nrows() {
            if observed_features[i] != convert(ba_state_linearizer::NO_FEATURE_FLAG) {
                residual_vector[i] = estimated_features[i] - observed_features[i];
            } else {
                residual_vector[i] = F::zero();
            }
            assert!(!residual_vector[i].is_nan());
        }
    }

    fn compute_jacobian_wrt_camera_extrinsics(
        camera: &C,
        state: &State<F, L, LANDMARK_PARAM_SIZE>,
        cam_idx: usize,
        point: &Point3<F>,
        i: usize,
        j: usize,
        jacobian: &mut DMatrix<F>,
    ) -> () {
        let transformation = state.to_se3(cam_idx);
        let transformed_point =
            transformation * Vector4::<F>::new(point[0], point[1], point[2], F::one());
        let lie_jacobian = left_jacobian_around_identity(&transformed_point.fixed_rows::<3>(0));

        let projection_jacobian = camera
            .get_jacobian_with_respect_to_position_in_camera_frame(
                &transformed_point.fixed_rows::<3>(0),
            )
            .expect("get_jacobian_with_respect_to_position_in_camera_frame failed!");
        let local_jacobian = projection_jacobian * lie_jacobian;

        jacobian
            .fixed_view_mut::<2, 6>(i, j)
            .copy_from(&local_jacobian);
    }

    fn compute_jacobian(
        state: &State<F, L, LANDMARK_PARAM_SIZE>,
        cameras: &Vec<&C>,
        jacobian: &mut DMatrix<F>,
    ) -> () {
        //cam
        let number_of_cam_params = CAMERA_PARAM_SIZE * state.n_cams;
        for cam_state_idx in (0..number_of_cam_params).step_by(CAMERA_PARAM_SIZE) {
            let cam_id = cam_state_idx / CAMERA_PARAM_SIZE;
            let camera = cameras[cam_id];
            let column = cam_state_idx;

            //landmark
            for point_id in 0..state.n_points {
                let point = state.get_landmarks()[point_id].get_euclidean_representation();
                let row = Self::get_feature_index_in_residual(cam_id, point_id, state.n_cams);

                Self::compute_jacobian_wrt_camera_extrinsics(
                    camera,
                    state,
                    cam_state_idx,
                    &point,
                    row,
                    column,
                    jacobian,
                );
            }
        }
    }

    pub fn compute_state_size(state: &State<F, L, LANDMARK_PARAM_SIZE>) -> usize {
        CAMERA_PARAM_SIZE * state.n_cams
    }

    pub fn solve(
        &self,
        state: &mut State<F, L, LANDMARK_PARAM_SIZE>,
        cameras: &Vec<&C>,
        observed_features: &DVector<F>,
        runtime_parameters: &RuntimeParameters<F>,
        abort_receiver: Option<&mpsc::Receiver<bool>>,
        done_transmission: Option<&mpsc::Sender<bool>>,
    ) -> Option<Vec<(Vec<[F; CAMERA_PARAM_SIZE]>, Vec<[F; LANDMARK_PARAM_SIZE]>)>> {
        self.optimizer.optimize(
            state,
            cameras,
            observed_features,
            runtime_parameters,
            abort_receiver,
            done_transmission,
        )
    }
}
