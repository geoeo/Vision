// THIS FILE SHOULD BE REFACTORED TO PROVIDE A SOLVER TEMPLATE!

extern crate nalgebra as na;

use na::{
    storage::Storage, DMatrix, DVector, Dynamic, Matrix, Matrix4, RowVector2, SMatrix, SVector,
    VecStorage, Vector3, Vector4, Vector6, U2, U4, U6, U9, Const, DimMin, DimMinimum
};
use std::boxed::Box;

use crate::features::geometry::point::Point;
use crate::image::Image;
use crate::numerics::{lie, loss::LossFunction};
use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::visual_odometry::dense_direct::{ RuntimeMemory,
    backproject_points, compute_full_jacobian, compute_image_gradients, compute_residuals,
    precompute_jacobians,norm,weight_jacobian_sparse,weight_residuals_sparse
};
use crate::odometry::{
    imu_odometry,
    imu_odometry::imu_delta::ImuDelta,
    imu_odometry::{weight_residuals, weight_jacobian},
};
use crate::pyramid::gd::{gd_octave::GDOctave, GDPyramid};
use crate::sensors::camera::{pinhole::Pinhole, Camera};
use crate::sensors::imu::imu_data_frame::ImuDataFrame;
use crate::{float, reconstruct_original_coordiantes, Float};

//type ResidualDim = U9;
//const RESIDUAL_DIM: usize = 9;
//type Identity = SMatrix<Float, RESIDUAL_DIM, RESIDUAL_DIM>;

pub fn run_trajectory(
    source_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>,
    target_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>,
    intensity_camera: &Pinhole,
    depth_camera: &Pinhole,
    imu_data_measurements: &Vec<ImuDataFrame>,
    bias_gyroscope: &Vector3<Float>,
    bias_accelerometer: &Vector3<Float>,
    gravity_body: &Vector3<Float>,
    runtime_parameters: &RuntimeParameters,
) -> Vec<Matrix4<Float>> {
    let mut runtime_memory_vector = RuntimeMemory::<9>::from_pyramid(&source_rgdb_pyramids[0]);
    source_rgdb_pyramids
        .iter()
        .zip(target_rgdb_pyramids.iter())
        .zip(imu_data_measurements.iter())
        .enumerate()
        .map(|(i, ((source, target), imu_data_measurement))| {
            run::<9>(
                i + 1,
                &source,
                &target,
                &mut runtime_memory_vector,
                intensity_camera,
                depth_camera,
                imu_data_measurement,
                bias_gyroscope,
                bias_accelerometer,
                gravity_body,
                runtime_parameters,
            )
        })
        .collect::<Vec<Matrix4<Float>>>()
}

pub fn run<const T: usize>(
    iteration: usize,
    source_rgdb_pyramid: &GDPyramid<GDOctave>,
    target_rgdb_pyramid: &GDPyramid<GDOctave>,
    runtime_memory_vector: &mut Vec<RuntimeMemory<9>>,
    intensity_camera: &Pinhole,
    depth_camera: &Pinhole,
    imu_data_measurement: &ImuDataFrame,
    bias_gyroscope: &Vector3<Float>,
    bias_accelerometer: &Vector3<Float>,
    gravity_body: &Vector3<Float>,
    runtime_parameters: &RuntimeParameters,
) -> Matrix4<Float> where  Const<T>: DimMin<Const<T>, Output = Const<T>> {
    let octave_count = source_rgdb_pyramid.octaves.len();

    assert_eq!(octave_count, runtime_parameters.taus.len());
    assert_eq!(octave_count, runtime_parameters.step_sizes.len());
    assert_eq!(octave_count, runtime_parameters.max_iterations.len());

    let depth_image = &source_rgdb_pyramid.depth_image;
    let mut mat_result = Matrix4::<Float>::identity();

    let (preintegrated_measurement, imu_covariance) = imu_odometry::pre_integration(
        imu_data_measurement,
        bias_gyroscope,
        bias_accelerometer,
        gravity_body,
    );

    for index in (0..octave_count).rev() {
        let result = estimate::<9>(
            &source_rgdb_pyramid.octaves[index],
            depth_image,
            &target_rgdb_pyramid.octaves[index],
            &mut runtime_memory_vector[index],
            index,
            &mat_result,
            intensity_camera,
            depth_camera,
            imu_data_measurement,
            &preintegrated_measurement,
            &imu_covariance,
            gravity_body,
            runtime_parameters,
        );
        mat_result = result.0;
        let solver_iterations = result.1;

        if runtime_parameters.show_octave_result {
            println!(
                "{}, est_transform: {}, solver iterations: {}",
                iteration, mat_result, solver_iterations
            );
        }
    }

    if runtime_parameters.show_octave_result {
        println!("final: est_transform: {}", mat_result);
    }

    mat_result
}

//TODO: buffer all debug strings and print at the end. Also the numeric matricies could be buffered per octave level
fn estimate<const T: usize>(
    source_octave: &GDOctave,
    source_depth_image_original: &Image,
    target_octave: &GDOctave,
    runtime_memory: &mut RuntimeMemory<T>,
    octave_index: usize,
    initial_guess_mat: &Matrix4<Float>,
    intensity_camera: &Pinhole,
    depth_camera: &Pinhole,
    imu_data_measurement: &ImuDataFrame,
    preintegrated_measurement: &ImuDelta,
    imu_covariance: &SMatrix<Float,T,T>,
    gravity_body: &Vector3<Float>,
    runtime_parameters: &RuntimeParameters,
) -> (Matrix4<Float>, usize) where  Const<T>: DimMin<Const<T>, Output = Const<T>> {
    let source_image = &source_octave.gray_images[0];
    let target_image = &target_octave.gray_images[0];
    let x_gradient_image = &target_octave.x_gradients[0];
    let y_gradient_image = &target_octave.y_gradients[0];
    let (rows, cols) = source_image.buffer.shape();
    let number_of_pixels = rows * cols;
    let number_of_pixels_float = number_of_pixels as Float;
    let mut percentage_of_valid_pixels = 100.0;
    let identity = SMatrix::<Float,T,T>::identity();

    runtime_memory.weights_vec.fill(1.0);
    runtime_memory.image_gradient_points.clear();
    runtime_memory.new_image_gradient_points.clear();

    let (backprojected_points, backprojected_points_flags) = backproject_points(
        &source_image.buffer,
        &source_depth_image_original.buffer,
        &depth_camera,
        octave_index,
    );
    let constant_jacobians = precompute_jacobians(
        &backprojected_points,
        &backprojected_points_flags,
        &intensity_camera,
    );

    let mut est_transform = initial_guess_mat.clone();

    compute_residuals(
        &target_image.buffer,
        &source_image.buffer,
        &backprojected_points,
        &backprojected_points_flags,
        &est_transform,
        &intensity_camera,
        &mut runtime_memory.residuals,
        &mut runtime_memory.image_gradient_points,
    );
    runtime_memory.residuals_unweighted.copy_from(&runtime_memory.residuals);
    // norm(
    //     &runtime_memory.residuals,
    //     &runtime_parameters.intensity_weighting_function,
    //     &mut runtime_memory.weights_vec,
    // );
    weight_residuals_sparse(&mut runtime_memory.residuals, &runtime_memory.weights_vec);

    let mut estimate = ImuDelta::empty();
    let delta_t = imu_data_measurement.gyro_ts[imu_data_measurement.gyro_ts.len() - 1]
        - imu_data_measurement.gyro_ts[0]; // Gyro has a higher sampling rate

    let mut imu_residuals = imu_odometry::generate_residual::<T>(&estimate, preintegrated_measurement);
    let mut imu_residuals_unweighted = imu_residuals.clone();

    let imu_weights = match imu_covariance.cholesky() {
        Some(v) => v.inverse(),
        None => {
            println!("Warning Cholesky failed for imu covariance");
            identity
        }
    };
    let imu_weight_l_upper = imu_weights
        .cholesky()
        .expect("Cholesky Decomp Failed!")
        .l()
        .transpose();
    let mut imu_jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);

    weight_residuals(&mut imu_residuals, &imu_weight_l_upper);
    weight_jacobian(&mut imu_jacobian, &imu_weight_l_upper);

    let mut cost = compute_cost(
        &runtime_memory.residuals,
        &imu_residuals,
        &runtime_parameters.loss_function,
    ); 

    let mut max_norm_delta = float::MAX;
    let mut delta_thresh = float::MIN;
    let mut delta_norm = float::MAX;
    let mut nu = 2.0;

    let mut mu: Option<Float> = match runtime_parameters.lm {
        true => None,
        false => Some(0.0),
    };
    let step = match runtime_parameters.lm {
        true => 1.0,
        false => runtime_parameters.step_sizes[octave_index],
    };
    let tau = runtime_parameters.taus[octave_index];
    let max_iterations = runtime_parameters.max_iterations[octave_index];

    compute_image_gradients(
        &x_gradient_image.buffer,
        &y_gradient_image.buffer,
        &runtime_memory.image_gradient_points,
        &mut runtime_memory.image_gradients,
    );
    compute_full_jacobian::<T>(
        &runtime_memory.image_gradients,
        &constant_jacobians,
        &mut runtime_memory.full_jacobian,
    );

    weight_jacobian_sparse(&mut runtime_memory.full_jacobian, &runtime_memory.weights_vec);

    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (cost.sqrt() > runtime_parameters.eps[octave_index]))
        || (runtime_parameters.lm
            && (delta_norm >= delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))
        && iteration_count < max_iterations
    {
        if runtime_parameters.debug {
            println!(
                "it: {}, rmse: {}, valid pixels: {}%",
                iteration_count,
                cost.sqrt(),
                percentage_of_valid_pixels
            );
        }

        let (delta, g, gain_ratio_denom, mu_val) = gauss_newton_step_with_loss(
            &runtime_memory.residuals,
            &imu_residuals,
            &runtime_memory.full_jacobian,
            &imu_jacobian,
            &identity,
            mu,
            tau,
            cost,
            &runtime_parameters.loss_function,
            &mut runtime_memory.rescaled_jacobian_target,
            &mut runtime_memory.rescaled_residual_target
        );
        mu = Some(mu_val);

        let pertb = step * delta;
        let new_estimate = estimate.add_pertb(&pertb);
        let new_est_transform =
            lie::exp(&pertb.fixed_rows::<3>(0), &pertb.fixed_rows::<3>(3)) * est_transform;

        runtime_memory.new_image_gradient_points.clear();
        compute_residuals(
            &target_image.buffer,
            &source_image.buffer,
            &backprojected_points,
            &backprojected_points_flags,
            &new_est_transform,
            &intensity_camera,
            &mut runtime_memory.new_residuals,
            &mut runtime_memory.new_image_gradient_points,
        );
        runtime_memory.new_residuals_unweighted.copy_from(&runtime_memory.new_residuals);
        norm(
            &runtime_memory.new_residuals,
            &runtime_parameters.intensity_weighting_function,
            &mut runtime_memory.weights_vec,
        );
        weight_residuals_sparse(&mut runtime_memory.new_residuals, &runtime_memory.weights_vec);
        percentage_of_valid_pixels =
            (runtime_memory.new_image_gradient_points.len() as Float / number_of_pixels_float) * 100.0;

        let mut imu_new_residuals =
            imu_odometry::generate_residual::<T>(&new_estimate, preintegrated_measurement);
        let imu_new_residuals_unweighted = imu_new_residuals.clone();
        weight_residuals(&mut imu_new_residuals, &imu_weight_l_upper);

        let new_cost = compute_cost(
            &runtime_memory.new_residuals,
            &imu_new_residuals,
            &runtime_parameters.loss_function,
        );
        let cost_diff = cost - new_cost;
        let gain_ratio = cost_diff / gain_ratio_denom;
        if runtime_parameters.debug {
            println!("{},{}", cost, new_cost);
        }
        if gain_ratio >= 0.0 || !runtime_parameters.lm {
            estimate = new_estimate;
            est_transform = new_est_transform;

            cost = new_cost;
            max_norm_delta = g.max();
            delta_norm = delta.norm();
            delta_thresh =
                runtime_parameters.delta_eps * (estimate.norm() + runtime_parameters.delta_eps);

            runtime_memory.image_gradient_points = runtime_memory.new_image_gradient_points.clone();
            runtime_memory.residuals.copy_from(&runtime_memory.new_residuals);
            runtime_memory.residuals_unweighted.copy_from(&runtime_memory.new_residuals_unweighted);

            imu_residuals.copy_from(&imu_new_residuals);
            imu_residuals_unweighted.copy_from(&imu_new_residuals_unweighted);

            compute_image_gradients(
                &x_gradient_image.buffer,
                &y_gradient_image.buffer,
                &runtime_memory.image_gradient_points,
                &mut runtime_memory.image_gradients,
            );
            compute_full_jacobian::<T>(
                &runtime_memory.image_gradients,
                &constant_jacobians,
                &mut runtime_memory.full_jacobian,
            );
            weight_jacobian_sparse(&mut runtime_memory.full_jacobian, &runtime_memory.weights_vec);

            imu_jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);
            weight_jacobian(&mut imu_jacobian, &imu_weight_l_upper);

            let v: Float = 1.0 / 3.0;
            mu = Some(mu.unwrap() * v.max(1.0 - (2.0 * gain_ratio - 1.0).powi(3)));
            nu = 2.0;
        } else {
            mu = Some(nu * mu.unwrap());
            nu *= 2.0;
        }

        iteration_count += 1;
    }

    (est_transform, iteration_count)
}

//TODO: potential for optimization. Maybe use less memory/matrices.
#[allow(non_snake_case)]
fn gauss_newton_step_with_loss<const T: usize>(
    residuals: &DVector<Float>, 
    imu_residuals: &SVector<Float, T>,
    jacobian: &Matrix<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>,
    imu_jacobian: &SMatrix<Float,T,T>,
    identity: &SMatrix<Float,T,T>,
    mu: Option<Float>,
    tau: Float,
    current_cost: Float,
    loss_function: &Box<dyn LossFunction>,
    rescaled_jacobian_target: &mut Matrix<
        Float,
        Dynamic,
        Const<T>,
        VecStorage<Float, Dynamic, Const<T>>,
    >,
    rescaled_residuals_target: &mut DVector<Float>
) -> (
    SVector<Float, T>,
    SVector<Float, T>,
    Float,
    Float,
)  where Const<T>: DimMin<Const<T>, Output = Const<T>> {
    let selected_root = loss_function.select_root(current_cost);
    let first_deriv_at_cost = loss_function.first_derivative_at_current(current_cost);
    let second_deriv_at_cost = loss_function.second_derivative_at_current(current_cost);
    let is_curvature_negative = second_deriv_at_cost * current_cost < -0.5 * first_deriv_at_cost;
    let approximate_gauss_newton_matrices = loss_function.approximate_gauss_newton_matrices();
    let (A, g) = match selected_root {
        root if root != 0.0 && approximate_gauss_newton_matrices => match is_curvature_negative {
            false => {
                let first_derivative_sqrt = loss_function
                    .first_derivative_at_current(current_cost)
                    .sqrt();
                let jacobian_factor = selected_root / current_cost;
                let residual_scale = first_derivative_sqrt / (1.0 - selected_root);
                let res_j = residuals.transpose() * jacobian;

                for i in 0..jacobian.nrows() {
                    rescaled_jacobian_target.row_mut(i).copy_from(
                        &(first_derivative_sqrt
                            * (jacobian.row(i) - (jacobian_factor * residuals[i] * res_j))),
                    );
                    rescaled_residuals_target[i] = residual_scale * residuals[i];
                }
                (
                    rescaled_jacobian_target.transpose()
                        * rescaled_jacobian_target
                            as &Matrix<
                                Float,
                                Dynamic,
                                Const<T>,
                                VecStorage<Float, Dynamic, Const<T>>,
                            >
                        + imu_jacobian.transpose()* imu_jacobian,
                    rescaled_jacobian_target.transpose()
                        * rescaled_residuals_target as &DVector<Float>
                        + imu_jacobian.transpose()* imu_residuals,
                )
            }
            _ => {

                (
                    jacobian.transpose()*first_deriv_at_cost*jacobian+2.0*second_deriv_at_cost*jacobian.transpose() * residuals*residuals.transpose() * jacobian + 
                    imu_jacobian.transpose()* imu_jacobian,
                    first_deriv_at_cost * jacobian.transpose() * residuals + imu_jacobian.transpose() * imu_residuals
                )
            }
        },
        _ => (
            jacobian.transpose() * jacobian + imu_jacobian.transpose() * imu_jacobian,
            jacobian.transpose() * residuals + imu_jacobian.transpose() * imu_residuals,
        ),
    };
    let mu_val = match mu {
        None => tau * A.diagonal().max(),
        Some(v) => v,
    };

    let decomp = (A + mu_val * identity).qr();
    let h = decomp.solve(&(-g)).expect("QR Solve Failed");
    let gain_ratio_denom = h.transpose() * (mu_val * h - g);
    (h, g, gain_ratio_denom[0], mu_val)
}

//TODO: make this more generic
fn compute_cost<const T: usize>(
    visual_residuals: &DVector<Float>,
    imu_residuals: &SVector<Float,T>,
    loss_function: &Box<dyn LossFunction>,
) -> Float {
    loss_function.cost((visual_residuals.transpose() * visual_residuals)[0]) + 
    (imu_residuals.transpose() * imu_residuals)[0]
}




