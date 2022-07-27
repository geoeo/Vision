// THIS FILE SHOULD BE REFACTORED TO PROVIDE A SOLVER TEMPLATE!

extern crate nalgebra as na;

use na::{
    DVector, Dynamic, Matrix, Matrix4,Isometry3,Rotation3, SMatrix, SVector,
    VecStorage, Vector3, Const, DimMin
};
use std::boxed::Box;

use crate::image::Image;
use crate::numerics::{lie, loss::LossFunction, max_norm, least_squares::{calc_weight_vec,weight_jacobian_sparse,weight_residuals_sparse,weight_jacobian,weight_residuals}};
use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::visual_odometry::dense_direct::{ RuntimeMemory,
    backproject_points, compute_full_jacobian, compute_image_gradients, compute_residuals, precompute_jacobians
};
use crate::odometry::{
    imu_odometry,
    imu_odometry::imu_delta::ImuDelta,
    imu_odometry::ImuCovariance,
    imu_odometry::bias::{BiasDelta,BiasPreintegrated},
    imu_odometry::bias
};
use crate::image::pyramid::gd::{gd_octave::GDOctave, GDPyramid};
use crate::sensors::camera::Camera;
use crate::sensors::imu::imu_data_frame::ImuDataFrame;
use crate::{float, Float};

const OBSERVATIONS_DIM: usize = 9;
const PARAMETERS_DIM: usize = 15; //With bias

pub fn run_trajectory<Cam: Camera>(
    source_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>,
    target_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>,
    intensity_camera: &Cam,
    depth_camera: &Cam,
    imu_data_measurements: &Vec<ImuDataFrame>,
    gravity_body: &Vector3<Float>,
    runtime_parameters: &RuntimeParameters,
) -> Vec<Isometry3<Float>> {
    let mut runtime_memory_vector = RuntimeMemory::<PARAMETERS_DIM>::from_pyramid(&source_rgdb_pyramids[0]);
    let mut bias_delta = BiasDelta::empty();
    source_rgdb_pyramids
        .iter()
        .zip(target_rgdb_pyramids.iter())
        .zip(imu_data_measurements.iter())
        .enumerate()
        .map(|(i, ((source, target), imu_data_measurement))| {
            let (transform_est, bias_update) = run::<Cam, PARAMETERS_DIM> (
                i + 1,
                &source,
                &target,
                &mut runtime_memory_vector,
                intensity_camera,
                depth_camera,
                imu_data_measurement,
                &bias_delta,
                gravity_body,
                runtime_parameters,
            );
            bias_delta = bias_delta.add_delta(&bias_update);
            let rotation = Rotation3::<Float>::from_matrix(&transform_est.fixed_slice::<3,3>(0,0).into_owned());
            Isometry3::<Float>::new(transform_est.fixed_slice::<3,1>(0,3).into_owned(),rotation.scaled_axis())
        })
        .collect::<Vec<Isometry3<Float>>>()
}

pub fn run<Cam: Camera, const C: usize>(
    iteration: usize,
    source_rgdb_pyramid: &GDPyramid<GDOctave>,
    target_rgdb_pyramid: &GDPyramid<GDOctave>,
    runtime_memory_vector: &mut Vec<RuntimeMemory<C>>,
    intensity_camera: &Cam,
    depth_camera: &Cam,
    imu_data_measurement: &ImuDataFrame,
    prev_bias_delta: &BiasDelta,
    gravity_body: &Vector3<Float>,
    runtime_parameters: &RuntimeParameters,
) -> (Matrix4<Float>, BiasDelta) where Const<C>: DimMin<Const<C>, Output = Const<C>> {
    let octave_count = source_rgdb_pyramid.octaves.len();

    assert_eq!(octave_count, runtime_parameters.taus.len());
    assert_eq!(octave_count, runtime_parameters.step_sizes.len());
    assert_eq!(octave_count, runtime_parameters.max_iterations.len());



    let imu_data_measurement_with_bias = imu_data_measurement.new_from_bias(prev_bias_delta);

    println!("bias a: {}",imu_data_measurement_with_bias.bias_a);
    println!("bias g: {}",imu_data_measurement_with_bias.bias_g);

    let depth_image = &source_rgdb_pyramid.depth_image;
    let mut mat_result = Matrix4::<Float>::identity();

    let (preintegrated_measurement, imu_covariance, preintegrated_bias) = imu_odometry::pre_integration(
        &imu_data_measurement_with_bias,
        gravity_body,
    );

    let mut bias_delta = BiasDelta::empty();
    for index in (0..octave_count).rev() {
        let result = estimate::<Cam, OBSERVATIONS_DIM, C>(
            &source_rgdb_pyramid.octaves[index],
            depth_image,
            &target_rgdb_pyramid.octaves[index],
            &mut runtime_memory_vector[index],
            index,
            &mat_result,
            &bias_delta,
            intensity_camera,
            depth_camera,
            imu_data_measurement,
            &preintegrated_measurement,
            &preintegrated_bias,
            &imu_covariance,
            gravity_body,
            runtime_parameters,
        );
        mat_result = result.0;
        let solver_iterations = result.1;
        bias_delta = result.2;

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

    (mat_result,bias_delta)
}

//TODO: buffer all debug strings and print at the end. Also the numeric matricies could be buffered per octave level
fn estimate<Cam: Camera, const R: usize, const C: usize>(
    source_octave: &GDOctave,
    source_depth_image_original: &Image,
    target_octave: &GDOctave,
    runtime_memory: &mut RuntimeMemory<C>,
    octave_index: usize,
    initial_transform_estimate: &Matrix4<Float>,
    initial_bias_estimate: &BiasDelta,
    intensity_camera: &Cam,
    depth_camera: &Cam,
    imu_data_measurement: &ImuDataFrame,
    preintegrated_measurement: &ImuDelta,
    preintegrated_bias: &BiasPreintegrated,
    imu_covariance: &ImuCovariance,
    gravity_body: &Vector3<Float>,
    runtime_parameters: &RuntimeParameters,
) -> (Matrix4<Float>, usize, BiasDelta) where Const<C>: DimMin<Const<C>, Output = Const<C>> {
    let source_image = &source_octave.gray_images[0];
    let target_image = &target_octave.gray_images[0];
    let x_gradient_image = &target_octave.x_gradients[0];
    let y_gradient_image = &target_octave.y_gradients[0];
    let (rows, cols) = source_image.buffer.shape();
    let number_of_pixels = rows * cols;
    let number_of_pixels_float = number_of_pixels as Float;
    let mut percentage_of_valid_pixels = 100.0;
    let identity = SMatrix::<Float,C,C>::identity();
    let mut imu_residuals_full = SVector::<Float,R>::zeros();
    let mut imu_jacobian_full = SMatrix::<Float,R,C>::zeros();

    runtime_memory.weights_vec.fill(1.0);
    runtime_memory.image_gradient_points.clear();
    runtime_memory.new_image_gradient_points.clear();

    let (backprojected_points, backprojected_points_flags) = backproject_points(
        &source_image.buffer,
        &source_depth_image_original.buffer,
        depth_camera,
        runtime_parameters.pyramid_scale,
        octave_index,
    );
    let constant_jacobians = precompute_jacobians(
        &backprojected_points,
        &backprojected_points_flags,
        intensity_camera,
    );

    let mut est_transform = initial_transform_estimate.clone();

    compute_residuals(
        &target_image.buffer,
        &source_image.buffer,
        &backprojected_points,
        &backprojected_points_flags,
        &est_transform,
        intensity_camera,
        &mut runtime_memory.residuals,
        &mut runtime_memory.image_gradient_points,
    );

    //TODO: check weighting
    let mut std: Option<Float> = runtime_parameters.intensity_weighting_function.estimate_standard_deviation(&runtime_memory.new_residuals);
    if std.is_some() {
        calc_weight_vec(
            &runtime_memory.new_residuals,
            std,
            &runtime_parameters.intensity_weighting_function,
            &mut runtime_memory.weights_vec,
        );
        weight_residuals_sparse(&mut runtime_memory.residuals, &runtime_memory.weights_vec);
    }



    let mut estimate = ImuDelta::empty();
    //let mut bias_estimate = BiasDelta::empty();
    let mut bias_estimate: BiasDelta = *initial_bias_estimate;
    let delta_t = imu_data_measurement.gyro_ts[imu_data_measurement.gyro_ts.len() - 1]
        - imu_data_measurement.gyro_ts[0]; // Gyro has a higher sampling rate

    let mut imu_residuals = imu_odometry::generate_residual(&estimate, preintegrated_measurement, &bias_estimate, preintegrated_bias);
    let mut imu_residuals_unweighted = imu_residuals.clone();

    
    let mut bias_a_residuals = bias::compute_residual(&bias_estimate.bias_a_delta, &preintegrated_bias.integrated_bias_a);
    let mut bias_g_residuals = bias::compute_residual(&bias_estimate.bias_g_delta, &preintegrated_bias.integrated_bias_g);

    let imu_weights = match imu_covariance.cholesky() {
        Some(v) => v.inverse(),
        None => {
            println!("Warning Cholesky failed for imu covariance");
            ImuCovariance::identity()
        }
    };
    let imu_weight_l_upper = imu_weights
        .cholesky()
        .expect("Cholesky Decomp Failed!")
        .l()
        .transpose();
    let mut imu_jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);

    weight_residuals::<_,9>(&mut imu_residuals, &imu_weight_l_upper);
    weight_jacobian::<_,9,9>(&mut imu_jacobian, &imu_weight_l_upper);

    let mut bias_jacobian = bias::genrate_residual_jacobian(&bias_estimate, preintegrated_bias, &imu_residuals);

    bias::weight_residual(&mut bias_a_residuals, &preintegrated_bias.bias_a_std);
    bias::weight_residual(&mut bias_g_residuals, &preintegrated_bias.bias_g_std);

    let mut cost = compute_cost(
        &runtime_memory.residuals,
        &imu_residuals,
        &bias_a_residuals,
        &bias_g_residuals,
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
    compute_full_jacobian::<C>(
        &runtime_memory.image_gradients,
        &constant_jacobians,
        &mut runtime_memory.full_jacobian,
    );

    if std.is_some() {
        weight_jacobian_sparse(&mut runtime_memory.full_jacobian, &runtime_memory.weights_vec);
    }


    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (cost.sqrt() > runtime_parameters.eps[octave_index]))
        || (runtime_parameters.lm
            && (delta_norm > delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))
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
        imu_residuals_full.fixed_rows_mut::<9>(0).copy_from(&imu_residuals);
        imu_jacobian_full.fixed_slice_mut::<9,9>(0,0).copy_from(&imu_jacobian);
        imu_jacobian_full.fixed_slice_mut::<9,6>(0,9).copy_from(&bias_jacobian);

        let (delta, g, gain_ratio_denom, mu_val) = gauss_newton_step_with_loss(
            &runtime_memory.residuals,
            &imu_residuals_full,
            &runtime_memory.full_jacobian,
            &imu_jacobian_full,
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
        let new_estimate = estimate.add_pertb(&pertb.fixed_rows::<9>(0));
        let new_bias_estimate = bias_estimate.add_pertb(&pertb.fixed_rows::<6>(9));

        let new_est_transform =
            lie::exp_se3(&pertb.fixed_rows::<3>(0), &pertb.fixed_rows::<3>(3)) * est_transform;

        runtime_memory.new_image_gradient_points.clear();
        compute_residuals(
            &target_image.buffer,
            &source_image.buffer,
            &backprojected_points,
            &backprojected_points_flags,
            &new_est_transform,
            intensity_camera,
            &mut runtime_memory.new_residuals,
            &mut runtime_memory.new_image_gradient_points,
        );
        std = runtime_parameters.intensity_weighting_function.estimate_standard_deviation(&runtime_memory.new_residuals);
        if std.is_some() {

            calc_weight_vec(
                &runtime_memory.new_residuals,
                std,
                &runtime_parameters.intensity_weighting_function,
                &mut runtime_memory.weights_vec,
            );
            weight_residuals_sparse(&mut runtime_memory.new_residuals, &runtime_memory.weights_vec);
        }
        
        percentage_of_valid_pixels =
            (runtime_memory.new_image_gradient_points.len() as Float / number_of_pixels_float) * 100.0;

        let mut imu_new_residuals =
            imu_odometry::generate_residual(&new_estimate, preintegrated_measurement, &bias_estimate, preintegrated_bias);
        let imu_new_residuals_unweighted = imu_new_residuals.clone();
        weight_residuals::<_,9>(&mut imu_new_residuals, &imu_weight_l_upper);

        let mut new_bias_a_residuals = bias::compute_residual(&new_bias_estimate.bias_a_delta, &preintegrated_bias.integrated_bias_a);
        let mut new_bias_g_residuals = bias::compute_residual(&new_bias_estimate.bias_g_delta, &preintegrated_bias.integrated_bias_g);

        bias::weight_residual(&mut new_bias_a_residuals, &preintegrated_bias.bias_a_std);
        bias::weight_residual(&mut new_bias_g_residuals, &preintegrated_bias.bias_g_std);

        let new_cost = compute_cost(
            &runtime_memory.new_residuals,
            &imu_new_residuals,
            &new_bias_a_residuals,
            &new_bias_g_residuals,
            &runtime_parameters.loss_function,
        );

        let cost_diff = cost - new_cost;
        let gain_ratio = cost_diff / gain_ratio_denom;
        if runtime_parameters.debug {
            println!("{},{}", cost, new_cost);
        }
        if gain_ratio >= 0.0 || !runtime_parameters.lm {
            let mut state_vector = SVector::<Float,R>::zeros();
            state_vector.fixed_rows_mut::<9>(0).copy_from(&estimate.state_vector());

            estimate = new_estimate;
            bias_estimate = new_bias_estimate;
            est_transform = new_est_transform;

            cost = new_cost;

            max_norm_delta = max_norm(&g);
            delta_norm = pertb.norm();
            delta_thresh =
                runtime_parameters.delta_eps * (state_vector.norm() + bias_estimate.norm() + runtime_parameters.delta_eps);

            runtime_memory.image_gradient_points = runtime_memory.new_image_gradient_points.clone();
            runtime_memory.residuals.copy_from(&runtime_memory.new_residuals);

            imu_residuals.copy_from(&imu_new_residuals);
            imu_residuals_unweighted.copy_from(&imu_new_residuals_unweighted);
            bias_a_residuals.copy_from(&new_bias_a_residuals);
            bias_g_residuals.copy_from(&new_bias_g_residuals);

            compute_image_gradients(
                &x_gradient_image.buffer,
                &y_gradient_image.buffer,
                &runtime_memory.image_gradient_points,
                &mut runtime_memory.image_gradients,
            );
            compute_full_jacobian::<C>(
                &runtime_memory.image_gradients,
                &constant_jacobians,
                &mut runtime_memory.full_jacobian,
            );

    

            if std.is_some() {
                weight_jacobian_sparse(&mut runtime_memory.full_jacobian, &runtime_memory.weights_vec);
            }


            imu_jacobian = imu_odometry::generate_jacobian(&estimate.rotation_lie(), delta_t);
            weight_jacobian::<_,9,9>(&mut imu_jacobian, &imu_weight_l_upper);
            bias_jacobian = bias::genrate_residual_jacobian(&bias_estimate, preintegrated_bias, &imu_residuals);

            let v: Float = 1.0 / 3.0;
            mu = Some(mu.unwrap() * v.max(1.0 - (2.0 * gain_ratio - 1.0).powi(3)));
            nu = 2.0;
        } else {
            mu = Some(nu * mu.unwrap());
            nu *= 2.0;
        }

        iteration_count += 1;
    }

    (est_transform, iteration_count, bias_estimate)
}

#[allow(non_snake_case)]
fn gauss_newton_step_with_loss<const R: usize, const C: usize>(
    residuals: &DVector<Float>, 
    imu_residuals: &SVector<Float, R>,
    jacobian: &Matrix<Float, Dynamic, Const<C>, VecStorage<Float, Dynamic, Const<C>>>,
    imu_jacobian: &SMatrix<Float,R,C>,
    identity: &SMatrix<Float,C,C>,
    mu: Option<Float>,
    tau: Float,
    current_cost: Float,
    loss_function: &Box<dyn LossFunction>,
    rescaled_jacobian_target: &mut Matrix<
        Float,
        Dynamic,
        Const<C>,
        VecStorage<Float, Dynamic, Const<C>>,
    >,
    rescaled_residuals_target: &mut DVector<Float>
) -> (
    SVector<Float, C>,
    SVector<Float, C>,
    Float,
    Float,
)  where Const<C>: DimMin<Const<C>, Output = Const<C>> {
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
                                Const<C>,
                                VecStorage<Float, Dynamic, Const<C>>,
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

//TODO: make this more generic -> rework with weight function and std
fn compute_cost<const T: usize>(
    visual_residuals: &DVector<Float>,
    imu_residuals: &SVector<Float,T>,
    bias_a_residuals: &Vector3<Float>,
    bias_g_residuals: &Vector3<Float>,
    loss_function: &Box<dyn LossFunction>,
) -> Float {
    loss_function.cost((visual_residuals.transpose() * visual_residuals)[0]) + 
    (imu_residuals.transpose() * imu_residuals)[0] + bias::compute_cost_for_weighted(bias_a_residuals, bias_g_residuals)
}




