extern crate nalgebra as na;

use na::{
    DVector, DimMin, Dynamic, Matrix, Matrix4, SMatrix, SVector,
    VecStorage,Const, Isometry3, Rotation3
};
use std::boxed::Box;

use crate::image::Image;
use crate::numerics::{lie, loss::LossFunction, max_norm, least_squares::{calc_weight_vec, weight_jacobian_sparse, weight_residuals_sparse, compute_cost}};
use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::odometry::visual_odometry::dense_direct::{
    RuntimeMemory,backproject_points, compute_full_jacobian, compute_image_gradients, compute_residuals, precompute_jacobians
};
use crate::image::pyramid::gd::{gd_octave::GDOctave, GDPyramid};
use crate::sensors::camera::Camera;
use crate::{float, Float};


pub fn run_trajectory<C>(
    source_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>,
    target_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>,
    intensity_camera: &C,
    depth_camera: &C,
    runtime_parameters: &RuntimeParameters<Float>,
) -> Vec<Isometry3<Float>> where C: Camera<Float> {
    let mut runtime_memory_vector = RuntimeMemory::<6>::from_pyramid(&source_rgdb_pyramids[0]);
    source_rgdb_pyramids
        .iter()
        .zip(target_rgdb_pyramids.iter())
        .enumerate()
        .map(|(i, (source, target))| {
            run(
                i + 1,
                &source,
                &target,
                &mut runtime_memory_vector,
                intensity_camera,
                depth_camera,
                runtime_parameters,
            )
        })
        .collect::<Vec<Isometry3<Float>>>()
}

pub fn run<C: Camera<Float>, const T: usize>(
    iteration: usize,
    source_rgdb_pyramid: &GDPyramid<GDOctave>,
    target_rgdb_pyramid: &GDPyramid<GDOctave>,
    runtime_memory_vector: &mut Vec<RuntimeMemory<T>>,
    intensity_camera: &C,
    depth_camera: &C,
    runtime_parameters: &RuntimeParameters<Float>,
) -> Isometry3<Float> where Const<T>: DimMin<Const<T>, Output = Const<T>> {
    let octave_count = source_rgdb_pyramid.octaves.len();

    assert_eq!(octave_count, runtime_parameters.taus.len());
    assert_eq!(octave_count, runtime_parameters.step_sizes.len());
    assert_eq!(octave_count, runtime_parameters.max_iterations.len());

    let depth_image = &source_rgdb_pyramid.depth_image;
    let mut mat_result = Matrix4::<Float>::identity();

    for index in (0..octave_count).rev() {
        let result = estimate(
            &source_rgdb_pyramid.octaves[index],
            depth_image,
            &target_rgdb_pyramid.octaves[index],
            &mut runtime_memory_vector[index],
            index,
            &mat_result,
            intensity_camera,
            depth_camera,
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

    let rotation = Rotation3::<Float>::from_matrix(&mat_result.fixed_slice::<3,3>(0,0).into_owned());
    Isometry3::<Float>::new(mat_result.fixed_slice::<3,1>(0,3).into_owned(),rotation.scaled_axis())
}

//TODO: buffer all debug strings and print at the end. Also the numeric matricies could be buffered per octave level
fn estimate<C : Camera<Float>, const T: usize>(
    source_octave: &GDOctave,
    source_depth_image_original: &Image,
    target_octave: &GDOctave,
    runtime_memory: &mut RuntimeMemory<T>,
    octave_index: usize,
    initial_guess_mat: &Matrix4<Float>,
    intensity_camera: &C,
    depth_camera: &C,
    runtime_parameters: &RuntimeParameters<Float>,
) -> (Matrix4<Float>, usize) where Const<T>: DimMin<Const<T>, Output = Const<T>> {
    let source_image = &source_octave.gray_images[0];
    let target_image = &target_octave.gray_images[0];
    let x_gradient_image = &target_octave.x_gradients[0];
    let y_gradient_image = &target_octave.y_gradients[0];
    let (rows, cols) = source_image.buffer.shape();
    let number_of_pixels = rows * cols;
    let number_of_pixels_float = number_of_pixels as Float;
    let mut percentage_of_valid_pixels = 100.0;

    runtime_memory.weights_vec.fill(1.0);
    runtime_memory.image_gradient_points.clear();
    runtime_memory.new_image_gradient_points.clear();


    let identity = SMatrix::<Float,T,T>::identity();
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

    let mut est_transform = initial_guess_mat.clone();

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

    let mut std = runtime_parameters.intensity_weighting_function.estimate_standard_deviation(&runtime_memory.new_residuals);
    if std.is_some() {
        calc_weight_vec(
            &runtime_memory.new_residuals,
            std,
            &runtime_parameters.intensity_weighting_function,
            &mut runtime_memory.weights_vec,
        );
        weight_residuals_sparse(&mut runtime_memory.new_residuals, &runtime_memory.weights_vec);
    }


    let mut cost = compute_cost(&runtime_memory.residuals, &runtime_parameters.intensity_weighting_function);

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


    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && cost.sqrt() > runtime_parameters.eps[octave_index])
        || (runtime_parameters.lm
            && delta_norm > delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps && cost.sqrt() > runtime_parameters.eps[octave_index]))
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
            &runtime_memory.full_jacobian,
            &identity,
            mu,
            tau,
            cost,
            &runtime_parameters.loss_function,
            &mut runtime_memory.rescaled_jacobian_target,
            &mut runtime_memory.rescaled_residual_target,
        );
        mu = Some(mu_val);

        let pertb = step * delta;
        let new_est_transform = lie::exp_se3(
            &pertb.fixed_slice::<3, 1>(0, 0),
            &pertb.fixed_slice::<3, 1>(3, 0),
        ) * est_transform;

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
        let new_cost = compute_cost(&runtime_memory.new_residuals, &runtime_parameters.intensity_weighting_function);
        let cost_diff = cost - new_cost;
        let gain_ratio = cost_diff / gain_ratio_denom;
        if runtime_parameters.debug {
            println!("{},{},{}", cost, new_cost,gain_ratio);
        }
        if gain_ratio >= 0.0 || !runtime_parameters.lm {
            let est_lie = lie::ln(&new_est_transform);
            est_transform = new_est_transform.clone();
            cost = new_cost;

            max_norm_delta = max_norm(&g); 
            delta_norm = pertb.norm(); 
            delta_thresh =
                runtime_parameters.delta_eps * (est_lie.norm() + runtime_parameters.delta_eps);

            runtime_memory.image_gradient_points = runtime_memory.new_image_gradient_points.clone();
            runtime_memory.residuals.copy_from(&runtime_memory.new_residuals);

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

            if std.is_some() {
                weight_jacobian_sparse(&mut runtime_memory.full_jacobian, &runtime_memory.weights_vec);
            }


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
    jacobian: &Matrix<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>,
    identity: &SMatrix<Float, T, T>,
    mu: Option<Float>,
    tau: Float,
    current_cost: Float,
    loss_function: &Box<dyn LossFunction>,
    rescaled_jacobian_target: &mut Matrix<
        Float,
        Dynamic,
        Const<T>,
        VecStorage<Float, Dynamic, Const<T>>>,
    rescaled_residuals_target: &mut DVector<Float>
) -> (
    SVector<Float, T>,
    SVector<Float, T>,
    Float,
    Float
) where Const<T>: DimMin<Const<T>, Output = Const<T>> {
    let selected_root = loss_function.select_root(current_cost);
    let first_deriv_at_cost = loss_function.first_derivative_at_current(current_cost);
    let second_deriv_at_cost = loss_function.second_derivative_at_current(current_cost);
    let is_curvature_negative = second_deriv_at_cost * current_cost < -0.5 * first_deriv_at_cost;

    let (A, g) = match selected_root {
        root if root != 0.0 => match is_curvature_negative {
            false => {
                let first_derivative_sqrt = first_deriv_at_cost.sqrt();
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
                            >,
                    rescaled_jacobian_target.transpose()
                        * rescaled_residuals_target as &DVector<Float>,
                )
            }
            _ => {
                (jacobian.transpose()*first_deriv_at_cost*jacobian+2.0*second_deriv_at_cost*jacobian.transpose() * residuals*residuals.transpose() * jacobian,first_deriv_at_cost * jacobian.transpose() * residuals)
            }
        },
        _ => (
            jacobian.transpose() * jacobian,
            jacobian.transpose() * residuals,
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


