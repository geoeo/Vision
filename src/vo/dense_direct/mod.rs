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
    let mut lie_result = Vector6::<Float>::zeros();
    let mut mat_result = Matrix4::<Float>::identity();
    
    for index in (0..source_rgdb_pyramid.octaves.len()).rev() {
        let result = estimate(&source_rgdb_pyramid.octaves[index],depth_image,&target_rgdb_pyramid.octaves[index],index,&lie_result,&mat_result,pinhole_camera,runtime_parameters);
        lie_result = result.0;
        mat_result = result.1;
    }

    mat_result
}

pub fn estimate(source_octave: &RGBDOctave, source_depth_image_original: &Image, target_octave: &RGBDOctave, octave_index: usize, initial_guess_lie: &Vector6<Float>,initial_guess_mat: &Matrix4<Float>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> (Vector6<Float>,Matrix4<Float>) {
    let source_image = &source_octave.gray_images[0];
    let target_image = &target_octave.gray_images[0];
    let x_gradient_image = &target_octave.x_gradients[0];
    let y_gradient_image = &target_octave.y_gradients[0];
    let (rows,cols) = source_image.buffer.shape();
    let number_of_pixels = rows*cols;
    let number_of_pixels_float = number_of_pixels as Float;

    let weights = DMatrix::<Float>::identity(rows,cols);
    let mut residuals = DVector::<Float>::from_element(number_of_pixels,float::MAX);
    let mut full_jacobian = Matrix::<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>::zeros(number_of_pixels); 
    let mut image_gradients =  Matrix::<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>::zeros(number_of_pixels);
    let mut image_gradient_points = Vec::<Point<usize>>::with_capacity(number_of_pixels);

    let backprojected_points = backproject_points(&source_image.buffer, &source_depth_image_original.buffer, &pinhole_camera, octave_index);
    let constant_jacobians = precompute_jacobians(&backprojected_points,&pinhole_camera);
    let mut est_transform = initial_guess_mat.clone();
    let mut est_lie = initial_guess_lie.clone();
    
    compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&est_transform, &pinhole_camera, &mut residuals,&mut image_gradient_points);
    compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);
    while cost(&residuals,&weights)/number_of_pixels_float >= runtime_parameters.eps {

        compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);
        compute_full_jacobian(&image_gradients,&constant_jacobians,&mut full_jacobian);
        let delta = gauss_newton_step(&residuals, &weights, &full_jacobian);
        est_lie += delta;
        est_transform = lie::exp(&est_lie.fixed_slice::<U3, U1>(0, 0),&est_lie.fixed_slice::<U3, U1>(3, 0));

        image_gradient_points.clear();
        compute_residuals(&target_image.buffer, &source_image.buffer, &backprojected_points,&est_transform, &pinhole_camera, &mut residuals,&mut image_gradient_points);
        compute_image_gradients(&x_gradient_image.buffer,&y_gradient_image.buffer,&image_gradient_points,&mut image_gradients);

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


fn compute_image_gradients(x_gradients: &DMatrix<Float>, y_gradients: &DMatrix<Float>,image_gradient_points: &Vec<Point<usize>> , target: &mut Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>) -> () {
    for (idx,point) in image_gradient_points.iter().enumerate() {
        let r = point.y;
        let c = point.x;
        let x_grad = x_gradients[(r,c)];
        let y_grad = y_gradients[(r,c)];
        target.set_row(idx,&RowVector2::new(x_grad,y_grad));
    }

}

fn backproject_points(source_image_buffer: &DMatrix<Float>,depth_image_buffer: &DMatrix<Float>,  pinhole_camera: &Pinhole, octave_index: usize) -> Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>> {
    let (rows,cols) = source_image_buffer.shape();
    let mut backproject_points =  Matrix::<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>::zeros(rows*cols);
    let number_of_points = rows*cols;

    for i in 0..number_of_points {
        let image_point =  linear_to_image_index(i, cols);
        let r = image_point.y;
        let c = image_point.x;
        let depth_sample = depth_image_buffer[reconstruct_original_coordiantes(c, r, octave_index as u32)];
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


        image_gradient_points.push(Point::<usize>::new(target_point_x,target_point_y));
        if source_point.y < rows && target_point_y < rows && 
           source_point.x < cols && target_point_x < cols {

            let source_sample = source_image_buffer[(source_point.y,source_point.x)];
            let target_sample = target_image_buffer[(target_point_y,target_point_x)];
            residual_target[i] = target_sample - source_sample;
           }
        else {
            residual_target[i] = 0.0;
        }

    }


}

//TODO: part of solver
#[allow(non_snake_case)]
fn gauss_newton_step(residuals: &DVector<Float>,weights: &DMatrix<Float>, motion_jacobian: &Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>) -> Vector6<Float> {
    let weights_sqr = weights*weights;
    let A = motion_jacobian.transpose()*&weights_sqr*motion_jacobian;
    let b = motion_jacobian.transpose()*&weights_sqr*residuals;
    let decomp = A.lu();
    decomp.solve(&b).expect("Linear resolution failed.")
}

fn cost(residuals: &DVector<Float>, weights: &DMatrix<Float>) -> Float {
    (residuals.transpose()*weights*weights*residuals)[0]
}

fn compute_full_jacobian(image_gradients: &Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>, const_jacobians: &Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>, target: &mut Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>) -> () {
    let number_of_elements = image_gradients.nrows();

    for i in 0..number_of_elements {
        let jacobian_i = image_gradients.row(i)*const_jacobians.fixed_slice::<U2,U6>(i*2,0);
        target.copy_from(&jacobian_i);
    }

}

