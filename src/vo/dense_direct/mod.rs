extern crate nalgebra as na;

use na::{U1,U2,U3,U4,U6,RowVector2,Vector4,Vector6,DVector,Matrix4,Matrix,Matrix6,DMatrix,Dynamic,VecStorage};

use crate::pyramid::rgbd::{RGBDPyramid,rgbd_octave::RGBDOctave};
use crate::camera::{Camera,pinhole::Pinhole};
use crate::vo::dense_direct::dense_direct_runtime_parameters::DenseDirectRuntimeParameters;
use crate::image::Image;
use crate::numerics::lie;
use crate::features::geometry::point::Point;
use crate::{Float,float,reconstruct_original_coordiantes};

pub mod dense_direct_runtime_parameters;

pub fn run_trajectory(source_rgdb_pyramids: &Vec<RGBDPyramid<RGBDOctave>>,target_rgdb_pyramids: &Vec<RGBDPyramid<RGBDOctave>>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> Vec<Matrix4<Float>> {
    source_rgdb_pyramids.iter().zip(target_rgdb_pyramids.iter()).map(|(source,target)| run(&source,&target, pinhole_camera,runtime_parameters)).collect::<Vec<Matrix4<Float>>>()
}

pub fn run(source_rgdb_pyramid: &RGBDPyramid<RGBDOctave>,target_rgdb_pyramid: &RGBDPyramid<RGBDOctave>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> Matrix4<Float> {

    let depth_image = &source_rgdb_pyramid.depth_image;
    let mut lie_result = Vector6::<Float>::zeros();
    let mut mat_result = Matrix4::<Float>::identity();
    
    for index in (0..source_rgdb_pyramid.octaves.len()).rev() {
        let result = estimate(&source_rgdb_pyramid.octaves[index],depth_image,&target_rgdb_pyramid.octaves[index],index,&lie_result,&mat_result,pinhole_camera,runtime_parameters);
        lie_result = result.0;
        mat_result = result.1;

        if runtime_parameters.show_octave_result {
            println!("est_transform: {}",mat_result);
        }

    }

    mat_result
}

fn estimate(source_octave: &RGBDOctave, source_depth_image_original: &Image, target_octave: &RGBDOctave, octave_index: usize, initial_guess_lie: &Vector6<Float>,initial_guess_mat: &Matrix4<Float>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> (Vector6<Float>,Matrix4<Float>) {
    let source_image = &source_octave.gray_images[0];
    let target_image = &target_octave.gray_images[0];
    let x_gradient_image = &target_octave.x_gradients[0];
    let y_gradient_image = &target_octave.y_gradients[0];
    let (rows,cols) = source_image.buffer.shape();
    let number_of_pixels = rows*cols;
    let number_of_pixels_float = number_of_pixels as Float;
    let mut percentage_of_valid_pixels = 100.0;

    let weights_vec = DVector::<Float>::from_element(number_of_pixels,1.0);
    let identity_6 = Matrix6::<Float>::identity();
    let mut residuals = DVector::<Float>::from_element(number_of_pixels,float::MAX);
    let mut new_residuals = DVector::<Float>::from_element(number_of_pixels,float::MAX);
    let mut full_jacobian = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut full_jacobian_weighted = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut image_gradients =  Matrix::<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>::zeros(number_of_pixels);
    let mut image_gradient_points = Vec::<Point<usize>>::with_capacity(number_of_pixels);
    let mut new_image_gradient_points = Vec::<Point<usize>>::with_capacity(number_of_pixels);
    let backprojected_points = backproject_points(&source_image.buffer, &source_depth_image_original.buffer, &pinhole_camera, octave_index);
    let constant_jacobians = precompute_jacobians(&backprojected_points,&pinhole_camera);

    let mut est_transform = initial_guess_mat.clone();
    let mut est_lie = initial_guess_lie.clone();

    //TODO: use mask to filter out of bounds pixels
    compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&est_transform, &pinhole_camera, &mut residuals,&mut image_gradient_points);
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
        false => runtime_parameters.step_size
    };

    compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);
    compute_full_jacobian(&image_gradients,&constant_jacobians,&mut full_jacobian);
    weight_jacobian(&mut full_jacobian_weighted,&full_jacobian, &weights_vec);
    weight_residuals(&mut residuals, &weights_vec); // We want the residual weighted by the square of the GN step


    let mut iteration_count = 0;
    while ((!runtime_parameters.lm && (avg_cost.sqrt() > runtime_parameters.eps)) || (runtime_parameters.lm && (delta_norm >= delta_thresh && max_norm_delta > runtime_parameters.max_norm_eps)))  && iteration_count < runtime_parameters.max_iterations  {
        if runtime_parameters.debug{
            println!("it: {}, avg_rmse: {}, valid pixels: {}%",iteration_count,avg_cost.sqrt(),percentage_of_valid_pixels);
        }

        let (delta,g,gain_ratio_denom, mu_val) = gauss_newton_step(&residuals, &full_jacobian, &full_jacobian_weighted, &identity_6, mu, runtime_parameters.tau);
        let new_est_lie = est_lie+ step*delta;

        let new_est_transform = lie::exp(&new_est_lie.fixed_slice::<U3, U1>(0, 0),&new_est_lie.fixed_slice::<U3, U1>(3, 0));
        mu = Some(mu_val);


        new_image_gradient_points.clear();
        compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&new_est_transform, &pinhole_camera,  &mut new_residuals,&mut new_image_gradient_points);
        weight_residuals(&mut new_residuals, &weights_vec);

        percentage_of_valid_pixels = (new_image_gradient_points.len() as Float/number_of_pixels_float) *100.0;
        let new_cost = compute_cost(&new_residuals);
        let cost_diff = cost-new_cost;
        let gain_ratio = cost_diff/gain_ratio_denom;
        if runtime_parameters.debug {
            //println!("{},{}",cost,new_cost);
        }
        if gain_ratio >= 0.0  || !runtime_parameters.lm {
            if runtime_parameters.debug{
                //println!("gain");
            }
            est_lie = new_est_lie.clone();
            est_transform = new_est_transform.clone();
            cost = new_cost;
            avg_cost = cost/new_image_gradient_points.len() as Float;

            max_norm_delta = g.max();
            delta_norm = delta.norm();
            delta_thresh = runtime_parameters.delta_eps*(est_lie.norm() + runtime_parameters.delta_eps);

            //println!("{},{}",max_norm_delta,delta_thresh);

            image_gradient_points = new_image_gradient_points.clone();
            residuals = new_residuals.clone();

            compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);
            compute_full_jacobian(&image_gradients,&constant_jacobians,&mut full_jacobian);
            weight_jacobian(&mut full_jacobian_weighted,&full_jacobian, &weights_vec);
            weight_residuals(&mut residuals, &weights_vec); // We want the residual weighted by the square of the GN step

            let v: Float = 1.0/3.0;
            mu = Some(mu.unwrap() * v.max(1.0-(2.0*gain_ratio-1.0).powi(3)));
            nu = 2.0;
        } else {
            mu = Some(nu*mu.unwrap());
            nu *= 2.0;


        
        }

        iteration_count += 1;

    }

    (est_lie,est_transform)
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

fn backproject_points(source_image_buffer: &DMatrix<Float>,depth_image_buffer: &DMatrix<Float>,  pinhole_camera: &Pinhole, octave_index: usize) -> Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>> {
    let (rows,cols) = source_image_buffer.shape();
    let mut backproject_points =  Matrix::<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>::zeros(rows*cols);
    let number_of_points = rows*cols;

    for i in 0..number_of_points {
        let image_point =  linear_to_image_index(i, cols);
        let r = image_point.y ;
        let c = image_point.x;
        let reconstruced_coordiantes = reconstruct_original_coordiantes(c, r, octave_index as u32);
        let depth_sample = depth_image_buffer[(reconstruced_coordiantes.1,reconstruced_coordiantes.0)];
        let backprojected_point = pinhole_camera.backproject(&Point::<Float>::new(c as Float,r as Float), depth_sample);
        backproject_points.set_column(image_to_linear_index(r,cols,c),&Vector4::<Float>::new(backprojected_point[0],backprojected_point[1],backprojected_point[2],1.0));
    }
    backproject_points
}

fn precompute_jacobians(backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>, pinhole_camera: &Pinhole) -> Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>> {
    let number_of_points = backprojected_points.ncols();
    let mut precomputed_jacobians = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(2*number_of_points);
    for i in 0..number_of_points {
        let point = backprojected_points.fixed_slice::<U3,U1>(0,i);
        let camera_jacobian = pinhole_camera.get_jacobian_with_respect_to_position(&point);
        let lie_jacobian = lie::jacobian_with_respect_to_transformation(&point);

        

        precomputed_jacobians.fixed_slice_mut::<U2,U6>(i*2,0).copy_from(&(camera_jacobian*lie_jacobian));
    }

    precomputed_jacobians
}

fn compute_residuals(target_image_buffer: &DMatrix<Float>,source_image_buffer: &DMatrix<Float>,backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>, est_transform: &Matrix4<Float>, pinhole_camera: &Pinhole, residual_target: &mut DVector<Float>,image_gradient_points: &mut Vec<Point<usize>>) -> () {
    let transformed_points = est_transform*backprojected_points;
    let number_of_points = transformed_points.ncols();
    let image_width = target_image_buffer.ncols();

    let (rows,cols) = source_image_buffer.shape();
    for i in 0..number_of_points {
        let source_point = linear_to_image_index(i,image_width);
        let target_point = pinhole_camera.project(&transformed_points.fixed_slice::<U3,U1>(0,i));
        let target_point_y = target_point.y.trunc() as usize;
        let target_point_x = target_point.x.trunc() as usize;


        if source_point.y < rows && target_point_y < rows && 
           source_point.x < cols && target_point_x < cols {
            image_gradient_points.push(Point::<usize>::new(target_point_x,target_point_y));
            let source_sample = source_image_buffer[(source_point.y,source_point.x)];
            let target_sample = target_image_buffer[(target_point_y,target_point_x)];
            residual_target[i] = target_sample - source_sample;
           }
        else {
            residual_target[i] = 0.0;
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

fn compute_t_dist_weight(residual: Float, variance: Float, t_dist_nu: Float) -> Float{
    (t_dist_nu + 1.0) / (t_dist_nu + residual.powi(2)/variance)
}

fn estimate_t_dist_variance(n: Float, residual: Float, t_dist_nu: Float) -> Float {
    panic!("not yet implemented")
}


fn weight_residuals( residual_target: &mut DVector<Float>, weights_vec: &DVector<Float>) -> () {
    residual_target.component_mul_assign(weights_vec);
}

fn weight_jacobian(jacobian_target: &mut Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>, weights_vec: &DVector<Float>) -> () {
    let size = weights_vec.len();
    for i in 0..size{
        let weighted_row = jacobian.row(i)*weights_vec[i].powi(2);
        jacobian_target.row_mut(i).copy_from(&weighted_row);
    }
}


#[allow(non_snake_case)]
fn gauss_newton_step(residuals_weighted: &DVector<Float>, jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,jacobian_weighted: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>,identity_6: &Matrix6<Float>, mu: Option<Float>, tau: Float) -> (Vector6<Float>,Vector6<Float>,Float,Float) {


    let A = jacobian.transpose()*jacobian_weighted;
    let mu_val = match mu {
        None => tau*A.diagonal().max(),
        Some(v) => v
    };

    let g = jacobian.transpose()*residuals_weighted;
    let decomp = (A+ mu_val*identity_6).lu();
    let h = decomp.solve(&(-g)).expect("Linear resolution failed.");
    let gain_ratio_denom = 0.5*h.transpose()*(mu_val*h-g);
    (h,g,gain_ratio_denom[0], mu_val)
}

fn compute_cost(residuals: &DVector<Float>) -> Float {
    (residuals.transpose()*residuals)[0]
}

