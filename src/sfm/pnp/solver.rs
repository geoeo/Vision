extern crate nalgebra as na;
extern crate num_traits;

use na::{DMatrix, DVector, Point3, Matrix, Dyn, U4, VecStorage};
use std::marker::{Send, Sync};
use std::sync::mpsc;

use crate::numerics::optimizer::gauss_newton::OptimizerGn;
use crate::sensors::camera::Camera;
use crate::sfm::runtime_parameters::RuntimeParameters;
use crate::sfm::state::{State, cam_state::CamState, landmark::Landmark};
use crate::GenericFloat;

pub struct Solver<
    F: Copy + GenericFloat,
    C: Camera<F> + Clone + Copy + 'static,
    L: Landmark<F, LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync + 'static,
    CS: CamState<F,C,CAMERA_PARAM_SIZE> + Copy + Clone + Send + Sync + 'static,
    const LANDMARK_PARAM_SIZE: usize,
    const CAMERA_PARAM_SIZE: usize
>
{
    optimizer: OptimizerGn<F,C, L,CS, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
}

impl<
        F: GenericFloat,
        C: Camera<F> + Clone + Copy + 'static,
        L: Landmark<F, LANDMARK_PARAM_SIZE> + Copy + Clone + Send + Sync + 'static,
        CP: CamState<F,C,CAMERA_PARAM_SIZE>  + Copy + Clone + Send + Sync + 'static,
        const LANDMARK_PARAM_SIZE: usize,
        const CAMERA_PARAM_SIZE: usize
    > Solver<F, C, L,CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>
{
    pub fn new() -> Solver<F,C, L,CP, LANDMARK_PARAM_SIZE,CAMERA_PARAM_SIZE> {
        Solver {
            optimizer: OptimizerGn::new(
                Box::new(Self::get_estimated_features),
                Box::new(Self::compute_residual),
                Box::new(Self::compute_jacobian),
                Box::new(Self::compute_state_size),
            ),
        }
    }

    /* *
     * In the format [f1_cam1, f1_cam2,...]
     * Some entries may be 0 since not all cams see all points
     * */
    fn get_estimated_features(
        state: &State<F,C,L, CP,LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
        _: &DVector<F>, //TODO: remove this
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
            let camera = state.get_camera(i); //@rework

            //TODO: use transform_into_other_camera_frame
            let transformed_points = pose * &position_world;
            for j in 0..n_points {
                let estimated_feature =
                    camera.project(&transformed_points.fixed_view::<3, 1>(0, j));

                let feat_id = Self::get_feature_index_in_residual(j);
                match estimated_feature {
                    Some(est) => {
                        estimated_features[feat_id] = est.x;
                        estimated_features[feat_id + 1] = est.y;
                    },
                    None => ()
                };
            }
        }
    }

    //TODO: ncams is unneccessary
    fn get_feature_index_in_residual(feature_id: usize) -> usize {
        2 * feature_id
    }

    fn compute_residual(
        estimated_features: &DVector<F>,
        observed_features: &DVector<F>,
        residual_vector: &mut DVector<F>,
    ) -> () {
        assert_eq!(residual_vector.nrows(), estimated_features.nrows());
        for i in 0..residual_vector.nrows() {
            residual_vector[i] = estimated_features[i] - observed_features[i];
            assert!(!residual_vector[i].is_nan());
        }
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

    fn compute_jacobian(
        state: &State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>,
        jacobian: &mut DMatrix<F>,
    ) -> () {
        //cam
        let number_of_cam_params = CAMERA_PARAM_SIZE * state.n_cams;
        for cam_num in (0..number_of_cam_params).step_by(CAMERA_PARAM_SIZE) {
            let column = cam_num;

            //landmark
            for point_id in 0..state.n_points {
                let point = state.get_landmarks()[point_id].get_euclidean_representation();
                let row = Self::get_feature_index_in_residual(point_id);

                Self::compute_jacobian_wrt_camera(
                    state,
                    &point,
                    row,
                    column,
                    jacobian,
                );
            }
        }
    }

    pub fn compute_state_size(state: &State<F, C, L, CP, LANDMARK_PARAM_SIZE, CAMERA_PARAM_SIZE>) -> usize {
        CAMERA_PARAM_SIZE * state.n_cams
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
