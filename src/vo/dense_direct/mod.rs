extern crate nalgebra as na;

use na::{U1,U2,U3,U4,U6,RowVector2,Vector2,Vector4,Vector6,DVector,Matrix4,Matrix,Matrix1x2,Matrix2x3,Matrix3x6,Matrix1x6,DMatrix,Dynamic,VecStorage};

use crate::pyramid::rgbd::{RGBDPyramid,rgbd_octave::RGBDOctave};
use crate::camera::{Camera,pinhole::Pinhole};
use crate::vo::dense_direct::dense_direct_runtime_parameters::DenseDirectRuntimeParameters;
use crate::image::Image;
use crate::numerics::lie;
use crate::features::geometry::point::Point;
use crate::{Float,float,reconstruct_original_coordiantes};

pub mod dense_direct_runtime_parameters;

pub fn run(source_rgdb_pyramid: &RGBDPyramid<RGBDOctave>,target_rgdb_pyramid: &RGBDPyramid<RGBDOctave>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> Matrix4<Float> {

    let depth_image = &source_rgdb_pyramid.depth_image;
    //assert!(depth_image.buffer.max() < 0.0);
    let mut lie_result = Vector6::<Float>::zeros();
    let mut mat_result = Matrix4::<Float>::identity();
    
    for index in (0..source_rgdb_pyramid.octaves.len()).rev() {
        let result = estimate(&source_rgdb_pyramid.octaves[index],depth_image,&target_rgdb_pyramid.octaves[index],index,&lie_result,&mat_result,pinhole_camera,runtime_parameters);
        lie_result = result.0;
        mat_result = result.1;
    }

    println!("est_transform: {}",mat_result);

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

    let weights_vec = DVector::<Float>::from_element(number_of_pixels,1.0);
    let mut residuals = DVector::<Float>::from_element(number_of_pixels,float::MAX);
    let mut full_jacobian = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut full_jacobian_weighted = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut image_gradients =  Matrix::<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>::zeros(number_of_pixels);
    let mut image_gradient_points = Vec::<Point<usize>>::with_capacity(number_of_pixels);
    let backprojected_points = backproject_points(&source_image.buffer, &source_depth_image_original.buffer, &pinhole_camera, octave_index, runtime_parameters);
    let constant_jacobians = precompute_jacobians(&backprojected_points,&pinhole_camera);

    let mut est_transform = initial_guess_mat.clone();
    let mut est_lie = initial_guess_lie.clone();

    compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&est_transform, &pinhole_camera,&runtime_parameters, &mut residuals,&mut image_gradient_points);
    weight_residuals(&mut residuals, &weights_vec);

    let mut iteration_count = 0;
    let mut avg_cost = cost(&residuals)/number_of_pixels_float;
    //TODO: step size via LM etc.
    let step_size = runtime_parameters.initial_step_size;

    while avg_cost >= runtime_parameters.eps && iteration_count < runtime_parameters.max_iterations {
        //TODO: put this in a log
        println!("it: {}, avg_cost: {}",iteration_count,avg_cost);

        compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&runtime_parameters,&mut image_gradients);
        compute_full_jacobian(&image_gradients,&constant_jacobians,&mut full_jacobian);
        weight_jacobian(&mut full_jacobian_weighted,&full_jacobian, &weights_vec);
        weight_residuals(&mut residuals, &weights_vec); // We want the residual weighted by the square of the GN step
        let delta = gauss_newton_step(&residuals, &full_jacobian, &full_jacobian_weighted);
        est_lie += delta*step_size;
        est_transform = lie::exp(&est_lie.fixed_slice::<U3, U1>(0, 0),&est_lie.fixed_slice::<U3, U1>(3, 0)); // check this


        image_gradient_points.clear();
        compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&est_transform, &pinhole_camera,& runtime_parameters,  &mut residuals,&mut image_gradient_points);
        weight_residuals(&mut residuals, &weights_vec);

        avg_cost = cost(&residuals)/number_of_pixels_float;
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


fn compute_image_gradients(x_gradients: &DMatrix<Float>, y_gradients: &DMatrix<Float>,image_gradient_points: &Vec<Point<usize>>, runtime_parameters: &DenseDirectRuntimeParameters, target: &mut Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>) -> () {
    for (idx,point) in image_gradient_points.iter().enumerate() {
        let r = point.y;
        let c = point.x;
        let factor = match runtime_parameters.invert_grad {
            true => -1.0,
            false => 1.0
        };
        if r < x_gradients.nrows() && c < x_gradients.ncols() {
            let x_grad = factor*x_gradients[(r,c)];
            let y_grad = factor*y_gradients[(r,c)];
            target.set_row(idx,&RowVector2::new(x_grad,y_grad));
        } else {
            target.set_row(idx,&RowVector2::new(0.0,0.0));
        }

    }

}

fn backproject_points(source_image_buffer: &DMatrix<Float>,depth_image_buffer: &DMatrix<Float>,  pinhole_camera: &Pinhole, octave_index: usize,  runtime_parameters: &DenseDirectRuntimeParameters) -> Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>> {
    let (rows,cols) = source_image_buffer.shape();
    let mut backproject_points =  Matrix::<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>::zeros(rows*cols);
    let number_of_points = rows*cols;

    for i in 0..number_of_points {
        let image_point =  linear_to_image_index(i, cols);
        let r = match runtime_parameters.invert_y {
            true => rows - 1 -  image_point.y,
            false => image_point.y
        };
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

fn compute_residuals(target_image_buffer: &DMatrix<Float>,source_image_buffer: &DMatrix<Float>,backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>, est_transform: &Matrix4<Float>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters, residual_target: &mut DVector<Float>,image_gradient_points: &mut Vec<Point<usize>>) -> () {
    let transformed_points = est_transform*backprojected_points;
    let number_of_points = transformed_points.ncols();
    let image_width = target_image_buffer.ncols();

    let (rows,cols) = source_image_buffer.shape();
    for i in 0..number_of_points {
        let source_point = linear_to_image_index(i,image_width);
        let target_point = pinhole_camera.project(&transformed_points.fixed_slice::<U3,U1>(0,i));
        let target_point_y = target_point.y.trunc() as usize;
        let target_point_x = target_point.x.trunc() as usize;

        image_gradient_points.push(Point::<usize>::new(target_point_x,target_point_y));
        if source_point.y < rows && target_point_y < rows && 
           source_point.x < cols && target_point_x < cols {

            let source_sample = source_image_buffer[(source_point.y,source_point.x)];
            let r = match runtime_parameters.invert_y {
                true => rows - 1 -  target_point_y,
                false => target_point_y
            };
            let target_sample = target_image_buffer[(r,target_point_x)];
            residual_target[i] = target_sample - source_sample;
            //residual_target[i] =  source_sample - target_sample;
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
fn gauss_newton_step(residuals_weighted: &DVector<Float>, jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>, jacobian_weighted: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>) -> Vector6<Float> {
    let A = jacobian.transpose()*jacobian_weighted;
    let b = -jacobian.transpose()*residuals_weighted;
    let decomp = A.lu();
    decomp.solve(&b).expect("Linear resolution failed.")
}

fn cost(residuals: &DVector<Float>) -> Float {
    (residuals.transpose()*residuals)[0]
}

