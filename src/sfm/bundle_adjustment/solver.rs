extern crate nalgebra as na;
extern crate num_traits;

use na::{convert, DMatrix, DVector, Dyn, Matrix, Point3, VecStorage, U4};
use std::boxed::Box;
use std::marker::{Send, Sync};
use std::sync::mpsc;

use crate::numerics::optimizer::gauss_newton_schur::OptimizerGnSchur;
use crate::sensors::camera::Camera;
use crate::sfm::{
    state::landmark::Landmark,
    runtime_parameters::RuntimeParameters,
    state::{ba_state_linearizer, State,cam_state::CamState},
};
use crate::GenericFloat;

pub struct Solver<
    F: GenericFloat,
    C: Camera<F> + Clone + Copy + 'static,
    L: Landmark<F, LANDMARK_PARAM_SIZE> + Send + Sync + 'static,
    CS: CamState<F,C, CAMERA_PARAM_SIZE> + Copy + Send + Sync + 'static,
    const LANDMARK_PARAM_SIZE: usize,
    const CAMERA_PARAM_SIZE: usize
>
{
    optimizer: OptimizerGnSchur<F, C, L,CS, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
}

impl<
        F: GenericFloat,
        C: Camera<F> + Clone + Copy + 'static,
        L: Landmark<F, LANDMARK_PARAM_SIZE> + Send + Sync + 'static,
        CP: CamState<F,C, CAMERA_PARAM_SIZE> + Copy + Send + Sync + 'static,
        const LANDMARK_PARAM_SIZE: usize,
        const CAMERA_PARAM_SIZE: usize
    > Solver<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>
{
    pub fn new() -> Solver<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE> {
        Solver {
            optimizer: OptimizerGnSchur::new(
                Box::new(Self::get_estimated_features),
                Box::new(Self::compute_residual),
                Box::new(Self::compute_jacobian),
                Box::new(Self::compute_state_size),
            ),
        }
    }

    fn get_feature_index_in_residual(cam_id: usize, feature_id: usize, n_cams: usize) -> usize {
        2 * (cam_id + feature_id * n_cams)
    }

    /* *
     * In the format [f1_cam1, f1_cam2,...]
     * Some entries may be 0 since not all cams see all points
     * */
    fn get_estimated_features(
        state: &State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
        observed_features: &DVector<F>,
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
            let cam_idx = CAMERA_PARAM_SIZE * i;
            let pose = state.to_isometry(cam_idx).to_matrix();
            let camera = state.get_camera(i);

            //TODO: use transform_into_other_camera_frame
            let transformed_points = pose * &position_world;
            for j in 0..n_points {
                let estimated_feature =
                    camera.project(&transformed_points.fixed_view::<3, 1>(0, j));

                let feat_id = Self::get_feature_index_in_residual(i, j, n_cams);
                // If at least one camera has no match, skip
                if !(observed_features[feat_id] == convert(ba_state_linearizer::NO_FEATURE_FLAG)
                    || observed_features[feat_id + 1] == convert(ba_state_linearizer::NO_FEATURE_FLAG)
                    || estimated_feature.is_none())
                {
                    let est = estimated_feature.unwrap();
                    estimated_features[feat_id] = est.x;
                    estimated_features[feat_id + 1] = est.y;
                }
            }
        }
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

    fn compute_jacobian_wrt_scene_points(
        state: &State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
        cam_idx: usize,
        point_idx: usize,
        i: usize,
        j: usize,
        jacobian: &mut DMatrix<F>,
    ) -> () {
        let point = state.get_landmarks()[point_idx].get_euclidean_representation();
        let jacobian_world = state.jacobian_wrt_landmarks(point_idx, cam_idx);
        let transformed_point = state.to_isometry(cam_idx).transform_point(&point).coords;
        let camera = state.get_camera(cam_idx/CAMERA_PARAM_SIZE); //@Rework
        let projection_jacobian = camera
            .get_jacobian_with_respect_to_position_in_camera_frame(
                &transformed_point,
            )
            .expect("get_jacobian_with_respect_to_position_in_camera_frame failed!");
        let local_jacobian = projection_jacobian * jacobian_world;

        jacobian
            .fixed_view_mut::<2, LANDMARK_PARAM_SIZE>(i, j)
            .copy_from(&local_jacobian.fixed_view::<2, LANDMARK_PARAM_SIZE>(0, 0));
    }

    fn compute_jacobian_wrt_camera(
        state: &State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
        point: &Point3<F>,
        landmark_index: usize,
        cam_idx: usize,
        jacobian: &mut DMatrix<F>,
    ) -> () {
        jacobian
            .fixed_view_mut::<2, CAMERA_PARAM_SIZE>(landmark_index, cam_idx)
            .copy_from(&state.jacobian_wrt_camera(point,cam_idx/CAMERA_PARAM_SIZE));
    }

    /**
     * Jacobian follow the structur of Sparse Bundle Adjustment of Lourakis et al.
        The [0, CAMERA_PARAM_SIZE * ncams) columns reprensent camera parameter values
        [CAMERA_PARAM_SIZE * ncams, .. ) represent landmark parameter values
     */
    fn compute_jacobian(
        state: &State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
        jacobian: &mut DMatrix<F>,
    ) -> () {
        //cam
        let number_of_cam_params = CAMERA_PARAM_SIZE * state.n_cams;
        for column_cam in (0..number_of_cam_params).step_by(CAMERA_PARAM_SIZE) {
            let cam_idx = column_cam / CAMERA_PARAM_SIZE;

            //landmark
            for point_id in 0..state.n_points {
                let point = state.get_landmarks()[point_id].get_euclidean_representation();

                let row = Self::get_feature_index_in_residual(cam_idx, point_id, state.n_cams);
                let column_landmark = number_of_cam_params + (LANDMARK_PARAM_SIZE * point_id);

                Self::compute_jacobian_wrt_camera(
                    state,
                    &point,
                    row,
                    column_cam,
                    jacobian,
                );

                Self::compute_jacobian_wrt_scene_points(
                    state,
                    column_cam,
                    point_id,
                    row,
                    column_landmark,
                    jacobian,
                );
            }
        }
    }

    pub fn compute_state_size(state: &State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>) -> usize {
        CAMERA_PARAM_SIZE * state.n_cams + LANDMARK_PARAM_SIZE * state.n_points
    }

    pub fn solve(
        &self,
        state: &mut State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
        observed_features: &DVector<F>,
        runtime_parameters: &RuntimeParameters<F>,
        abort_receiver: Option<&mpsc::Receiver<bool>>,
        done_transmission: Option<&mpsc::Sender<bool>>,
    ) -> Option<Vec<State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>>> {
        self.optimizer.optimize(
            state,
            observed_features,
            runtime_parameters,
            abort_receiver,
            done_transmission,
        )
    }
}
