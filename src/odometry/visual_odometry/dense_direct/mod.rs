extern crate nalgebra as na;

use na::{U1,U2,U3,U4,U6,RowVector2,Vector4,Vector6,DVector,Matrix4,Matrix,Matrix6,DMatrix,Dynamic,VecStorage};
use std::boxed::Box;

use crate::pyramid::gd::{GDPyramid,gd_octave::GDOctave};
use crate::sensors::camera::{Camera,pinhole::Pinhole};
use crate::odometry::runtime_parameters::RuntimeParameters;
use crate::image::Image;
use crate::numerics::{lie,loss::LossFunction};
use crate::features::geometry::point::Point;
use crate::{Float,float,reconstruct_original_coordiantes};

pub fn run_trajectory(source_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>,target_rgdb_pyramids: &Vec<GDPyramid<GDOctave>>, intensity_camera: &Pinhole, depth_camera: &Pinhole , runtime_parameters: &RuntimeParameters) -> Vec<Matrix4<Float>> {
    source_rgdb_pyramids.iter().zip(target_rgdb_pyramids.iter()).enumerate().map(|(i,(source,target))| run(i+1,&source,&target, intensity_camera,depth_camera,runtime_parameters)).collect::<Vec<Matrix4<Float>>>()
}

pub fn run(iteration: usize, source_rgdb_pyramid: &GDPyramid<GDOctave>,target_rgdb_pyramid: &GDPyramid<GDOctave>, intensity_camera: &Pinhole,depth_camera: &Pinhole, runtime_parameters: &RuntimeParameters) -> Matrix4<Float> {
    let octave_count = source_rgdb_pyramid.octaves.len();

    assert_eq!(octave_count,runtime_parameters.taus.len());
    assert_eq!(octave_count,runtime_parameters.step_sizes.len());
    assert_eq!(octave_count,runtime_parameters.max_iterations.len());

    let depth_image = &source_rgdb_pyramid.depth_image;
    let mut lie_result = Vector6::<Float>::zeros();
    let mut mat_result = Matrix4::<Float>::identity();
    
    for index in (0..octave_count).rev() {

        let result = estimate(&source_rgdb_pyramid.octaves[index],depth_image,&target_rgdb_pyramid.octaves[index],index,&lie_result,&mat_result,intensity_camera,depth_camera,runtime_parameters);
        lie_result = result.0;
        mat_result = result.1;
        let solver_iterations = result.2;

        if runtime_parameters.show_octave_result {
            println!("{}, est_transform: {}, solver iterations: {}",iteration,mat_result, solver_iterations);
        }

    }

    if runtime_parameters.show_octave_result {
        println!("final: est_transform: {}",mat_result);
    }

    mat_result
}

//TODO: buffer all debug strings and print at the end. Also the numeric matricies could be buffered per octave level
fn estimate(source_octave: &GDOctave, source_depth_image_original: &Image, target_octave: &GDOctave, octave_index: usize, initial_guess_lie: &Vector6<Float>,initial_guess_mat: &Matrix4<Float>, intensity_camera: &Pinhole, depth_camera: &Pinhole, runtime_parameters: &RuntimeParameters) -> (Vector6<Float>,Matrix4<Float>, usize) {
    let source_image = &source_octave.gray_images[0];
    let target_image = &target_octave.gray_images[0];
    let x_gradient_image = &target_octave.x_gradients[0];
    let y_gradient_image = &target_octave.y_gradients[0];
    let (rows,cols) = source_image.buffer.shape();
    let number_of_pixels = rows*cols;
    let number_of_pixels_float = number_of_pixels as Float;
    let mut percentage_of_valid_pixels = 100.0;

    let mut weights_vec = DVector::<Float>::from_element(number_of_pixels,1.0);
    let identity_6 = Matrix6::<Float>::identity();
    let mut residuals = DVector::<Float>::from_element(number_of_pixels,float::MAX);
    let mut new_residuals = DVector::<Float>::from_element(number_of_pixels,float::MAX);
    let mut full_jacobian = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut full_jacobian_weighted = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut image_gradients =  Matrix::<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>::zeros(number_of_pixels);
    let mut image_gradient_points = Vec::<Point<usize>>::with_capacity(number_of_pixels);
    let mut new_image_gradient_points = Vec::<Point<usize>>::with_capacity(number_of_pixels);
    let mut rescaled_jacobian_target = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut rescaled_residual_target =  DVector::<Float>::zeros(number_of_pixels);
    let (backprojected_points,backprojected_points_flags) = backproject_points(&source_image.buffer, &source_depth_image_original.buffer, &depth_camera, octave_index);
    let constant_jacobians = precompute_jacobians(&backprojected_points,&backprojected_points_flags,&intensity_camera);

    let mut est_transform = initial_guess_mat.clone();
    let mut est_lie = initial_guess_lie.clone();

    compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&backprojected_points_flags,&est_transform, &intensity_camera, &mut residuals,&mut image_gradient_points);
    weight_residuals(&mut residuals, &weights_vec);
    let mut cost = compute_cost(&residuals);
    let mut avg_cost = cost/image_gradient_points.len() as Float;

    let mut max_norm_delta = float::MAX;
    let mut delta_thresh = float::MIN;
    let mut delta_norm = float::MAX;
    let mut nu = 2.0;

    let mut mu: Option<Float> = match runtime_parameters.lm {
        true => None,
        false => Some(0.0)
    };
    let step = match runtime_parameters.lm {
        true => 1.0,
        false => runtime_parameters.step_sizes[octave_index]
    };
    let tau = runtime_parameters.taus[octave_index];
    let max_iterations = runtime_parameters.max_iterations[octave_index];
    

    compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);
    compute_full_jacobian(&image_gradients,&constant_jacobians,&mut full_jacobian);
    weight_jacobian(&mut full_jacobian_weighted,&full_jacobian, &weights_vec);
    

    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (avg_cost.sqrt() > runtime_parameters.eps[octave_index])) || (runtime_parameters.lm && (delta_norm >= delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))  && iteration_count < max_iterations  {
        if runtime_parameters.debug{
            println!("it: {}, avg_rmse: {}, valid pixels: {}%",iteration_count,avg_cost.sqrt(),percentage_of_valid_pixels);
        }

        let (delta,g,gain_ratio_denom, mu_val) 
            = gauss_newton_step_with_loss(&residuals, &full_jacobian_weighted, &identity_6, mu, tau, cost, & runtime_parameters.loss_function, &mut rescaled_jacobian_target,&mut rescaled_residual_target);
        mu = Some(mu_val);


        //let new_est_lie = est_lie+ step*delta;
        //let new_est_transform = lie::exp(&new_est_lie.fixed_rows::<U3>(0),&new_est_lie.fixed_rows::<U3>(3));

        let pertb = step*delta;
        let new_est_transform = lie::exp(&pertb.fixed_rows::<U3>(0),&pertb.fixed_rows::<U3>(3))*est_transform;


        new_image_gradient_points.clear();
        compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&backprojected_points_flags,&new_est_transform, &intensity_camera , &mut new_residuals,&mut new_image_gradient_points);
        
        
        if runtime_parameters.weighting {
            compute_t_dist_weights(&new_residuals,&mut weights_vec,new_image_gradient_points.len() as Float,5.0,20,1e-10);
        }
        weight_residuals(&mut new_residuals, &weights_vec);

        percentage_of_valid_pixels = (new_image_gradient_points.len() as Float/number_of_pixels_float) *100.0;
        let new_cost = compute_cost(&new_residuals);
        let cost_diff = cost-new_cost;
        let gain_ratio = cost_diff/gain_ratio_denom;
        if runtime_parameters.debug {
            println!("{},{}",cost,new_cost);
        }
        if gain_ratio >= 0.0  || !runtime_parameters.lm {
            //est_lie = new_est_lie.clone();
            est_lie = lie::ln(&new_est_transform);
            est_transform = new_est_transform.clone();
            cost = new_cost;
            avg_cost = cost/new_image_gradient_points.len() as Float;

            max_norm_delta = g.max();
            delta_norm = delta.norm();
            delta_thresh = runtime_parameters.delta_eps*(est_lie.norm() + runtime_parameters.delta_eps);

            image_gradient_points = new_image_gradient_points.clone();
            residuals = new_residuals.clone();

            compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);
            compute_full_jacobian(&image_gradients,&constant_jacobians,&mut full_jacobian);
            weight_jacobian(&mut full_jacobian_weighted, &full_jacobian, &weights_vec);

            let v: Float = 1.0/3.0;
            mu = Some(mu.unwrap() * v.max(1.0-(2.0*gain_ratio-1.0).powi(3)));
            nu = 2.0;
        } else {

            mu = Some(nu*mu.unwrap());
            nu *= 2.0;
        }

        iteration_count += 1;

    }

    (est_lie,est_transform,iteration_count)
}

fn image_to_linear_index(r: usize, cols: usize, c: usize) -> usize {
    r*cols+c
}

fn linear_to_image_index(idx: usize, cols: usize) -> Point<usize> {
    let x = (idx as Float % cols as Float).trunc() as usize;
    let y =  ((idx  as Float) / (cols as Float)).trunc() as usize;
    Point::<usize>::new(x ,y)
}

fn compute_image_gradients(x_gradients: &DMatrix<Float>, y_gradients: &DMatrix<Float>,image_gradient_points: &Vec<Point<usize>>, target: &mut Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>) -> () {
    for (idx,point) in image_gradient_points.iter().enumerate() {
        let r = point.y;
        let c = point.x;
        if r < x_gradients.nrows() && c < x_gradients.ncols() {
            let x_grad = x_gradients[(r,c)];
            let y_grad = y_gradients[(r,c)];
            target.set_row(idx,&RowVector2::new(x_grad,y_grad));
        } else {
            target.set_row(idx,&RowVector2::new(0.0,0.0));
        }

    }

}

fn backproject_points(source_image_buffer: &DMatrix<Float>,depth_image_buffer: &DMatrix<Float>,  pinhole_camera: &Pinhole, octave_index: usize) -> (Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>, Vec<bool>) {
    let (rows,cols) = source_image_buffer.shape();
    let mut backproject_points =  Matrix::<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>::zeros(rows*cols);
    let mut backproject_points_flags =  vec![false;rows*cols];
    let number_of_points = rows*cols;

    for i in 0..number_of_points {
        let image_point =  linear_to_image_index(i, cols);
        let r = image_point.y;
        let c = image_point.x;
        let reconstruced_coordiantes = reconstruct_original_coordiantes(c, r, octave_index as u32);
        let depth_sample = depth_image_buffer[(reconstruced_coordiantes.1,reconstruced_coordiantes.0)];


        if depth_sample != 0.0 {
            let backprojected_point = pinhole_camera.backproject(&Point::<Float>::new(c as Float + 0.5,r as Float + 0.5), depth_sample); //TODO: inverse depth
            backproject_points.set_column(image_to_linear_index(r,cols,c),&Vector4::<Float>::new(backprojected_point[0],backprojected_point[1],backprojected_point[2],1.0));
            backproject_points_flags[image_to_linear_index(r,cols,c)] = true;
        } else {
            backproject_points.set_column(image_to_linear_index(r,cols,c),&Vector4::<Float>::new(0.0,0.0,0.0,1.0));
        }
    }
    (backproject_points,backproject_points_flags)
}

fn precompute_jacobians(backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>,backprojected_points_flags: &Vec<bool>, pinhole_camera: &Pinhole) -> Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>> {
    let number_of_points = backprojected_points.ncols();
    let mut precomputed_jacobians = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(2*number_of_points);

    for i in 0..number_of_points {
        if backprojected_points_flags[i] {
            let point = backprojected_points.fixed_slice::<U3,U1>(0,i);
            let camera_jacobian = pinhole_camera.get_jacobian_with_respect_to_position(&point);
            let lie_jacobian = lie::left_jacobian_around_identity(&point);
            precomputed_jacobians.fixed_slice_mut::<U2,U6>(i*2,0).copy_from(&(camera_jacobian*lie_jacobian));
        }

        

    }

    precomputed_jacobians
}

fn compute_residuals(target_image_buffer: &DMatrix<Float>,source_image_buffer: &DMatrix<Float>,backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>,backprojected_points_flags: &Vec<bool>, est_transform: &Matrix4<Float>, pinhole_camera: &Pinhole, residual_target: &mut DVector<Float>, image_gradient_points: &mut Vec<Point<usize>>) -> () {
    let transformed_points = est_transform*backprojected_points;
    let number_of_points = transformed_points.ncols();
    let image_width = target_image_buffer.ncols();

    let (rows,cols) = source_image_buffer.shape();
    for i in 0..number_of_points {
        let source_point = linear_to_image_index(i,image_width);
        let target_point = pinhole_camera.project(&transformed_points.fixed_slice::<U3,U1>(0,i)); //TODO: inverse depth
        let target_point_y = target_point.y.trunc() as usize;
        let target_point_x = target_point.x.trunc() as usize;
        let target_linear_idx = image_to_linear_index(target_point_y, image_width, target_point_x);

        // We only want to compare pixels where both values are valid i.e. has depth estimate
        if source_point.y < rows && target_point_y < rows && 
           source_point.x < cols && target_point_x < cols && 
           backprojected_points_flags[i] &&
           backprojected_points_flags[target_linear_idx]{ 
            image_gradient_points.push(Point::<usize>::new(target_point_x,target_point_y));
            let source_sample = source_image_buffer[(source_point.y,source_point.x)];
            let target_sample = target_image_buffer[(target_point_y,target_point_x)];
            residual_target[i] = target_sample - source_sample;
           }
        else {
            residual_target[i] = 0.0; 
            //residual_target[i] = 1.0; 
        }

    }


}


fn compute_full_jacobian(image_gradients: &Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>, const_jacobians: &Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>, target: &mut Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>) -> () {
    let number_of_elements = image_gradients.nrows();

    for i in 0..number_of_elements {
        let jacobian_i = image_gradients.row(i)*const_jacobians.fixed_slice::<U2,U6>(i*2,0);
        target.fixed_slice_mut::<U1,U6>(i,0).copy_from(&jacobian_i);

    }

}



//TODO: part of solver

fn compute_t_dist_weights(residuals: &DVector<Float>, weights_vec: &mut DVector<Float>, n: Float, t_dist_nu: Float, max_it: usize, eps: Float) -> () {

    let variance = estimate_t_dist_variance(n,residuals,t_dist_nu,max_it,eps);
    for i in 0..residuals.len() {
        let res = residuals[i];
        weights_vec[i] = compute_t_dist_weight(res,variance,t_dist_nu);
    }
    
}

fn compute_t_dist_weight(residual: Float, variance: Float, t_dist_nu: Float) -> Float{
    (t_dist_nu + 1.0) / (t_dist_nu + residual.powi(2)/variance)
}

fn estimate_t_dist_variance(n: Float, residuals: &DVector<Float>, t_dist_nu: Float, max_it: usize, eps: Float) -> Float {
    let mut it = 0;
    let mut err = float::MAX;
    let mut variance = float::MAX; 

    while it < max_it && err > eps {
        let mut acc = 0.0;
        for r in residuals {
            if *r == 0.0 {
                continue;
            }
            let r_sqrd = r.powi(2);
            acc += r_sqrd*(t_dist_nu +1.0)/(t_dist_nu + r_sqrd/variance);
        }

        let var_old = variance;
        variance = acc/n;
        err = (variance-var_old).abs();
        it += 1;
    }

    variance
}


fn weight_residuals( residual_target: &mut DVector<Float>, weights_vec: &DVector<Float>) -> () {
    residual_target.component_mul_assign(weights_vec);
}

fn weight_jacobian(jacobian_target: &mut Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>, weights_vec: &DVector<Float>) -> () {
    let size = weights_vec.len();
    for i in 0..size{
        let weighted_row = jacobian.row(i)*weights_vec[i];
        jacobian_target.row_mut(i).copy_from(&weighted_row);
    }
}


#[allow(non_snake_case)]
fn gauss_newton_step(residuals_weighted: &DVector<Float>, jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,identity_6: &Matrix6<Float>, mu: Option<Float>, tau: Float) -> (Vector6<Float>,Vector6<Float>,Float,Float) {
    let A = jacobian.transpose()*jacobian;
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let g = jacobian.transpose()*residuals_weighted;
    let decomp = (A+ mu_val*identity_6).lu();
    let h = decomp.solve(&(-g)).expect("Linear resolution failed.");
    let gain_ratio_denom = h.transpose()*(mu_val*h-g);
    (h,g,gain_ratio_denom[0], mu_val)
}

//TODO: potential for optimization. Maybe use less memory/matrices. 
#[allow(non_snake_case)]
fn gauss_newton_step_with_loss(residuals: &DVector<Float>, 
    jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,
    identity_6: &Matrix6<Float>,
     mu: Option<Float>, 
     tau: Float, 
     current_cost: Float, 
     loss_function: &Box<dyn LossFunction>,
      rescaled_jacobian_target: &mut Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>, 
      rescaled_residuals_target: &mut DVector<Float>) -> (Vector6<Float>,Vector6<Float>,Float,Float) {

    let selected_root = loss_function.select_root(current_cost);
    let (A,g) =  match selected_root {

        root if root != 0.0 => {
            let first_derivative_sqrt = loss_function.first_derivative_at_current(current_cost).sqrt();
            let jacobian_factor = selected_root/current_cost;
            let residual_scale = first_derivative_sqrt/(1.0-selected_root);
            let res_j = residuals.transpose()*jacobian;
            for i in 0..jacobian.nrows(){
                rescaled_jacobian_target.row_mut(i).copy_from(&(first_derivative_sqrt*(jacobian.row(i) - (jacobian_factor*residuals[i]*res_j))));
                rescaled_residuals_target[i] = residual_scale*residuals[i];
            }
            (rescaled_jacobian_target.transpose()*rescaled_jacobian_target as &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,rescaled_jacobian_target.transpose()*rescaled_residuals_target as &DVector<Float>)
        },
        _ => (jacobian.transpose()*jacobian,jacobian.transpose()*residuals)
    };
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let decomp = (A+ mu_val*identity_6).lu();
    let h = decomp.solve(&(-g)).expect("Linear resolution failed.");
    let gain_ratio_denom = h.transpose()*(mu_val*h-g);
    (h,g,gain_ratio_denom[0], mu_val)
}


fn compute_cost(residuals: &DVector<Float>) -> Float {
    (residuals.transpose()*residuals)[0]
}

