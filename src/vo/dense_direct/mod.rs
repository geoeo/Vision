extern crate nalgebra as na;

use na::{U1,U2,U3,U4,U6,Vector2,Vector4,Vector6,DVector,Matrix4,Matrix,Matrix1x2,Matrix2x3,Matrix3x6,Matrix1x6,DMatrix,Dynamic,VecStorage};

use crate::pyramid::rgbd::{RGBDPyramid,rgbd_octave::RGBDOctave};
use crate::camera::{Camera,pinhole::Pinhole};
use crate::vo::dense_direct::dense_direct_runtime_parameters::DenseDirectRuntimeParameters;
use crate::image::Image;
use crate::numerics::lie;
use crate::features::geometry::point::Point;
use crate::{Float,reconstruct_original_coordiantes};

pub mod dense_direct_runtime_parameters;

pub fn run(source_rgdb_pyramid: &RGBDPyramid<RGBDOctave>,target_rgdb_pyramid: &RGBDPyramid<RGBDOctave>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> Matrix4<Float> {


    let depth_image = &source_rgdb_pyramid.depth_image;

    let mut estimation_result = Vector6::<Float>::zeros();
    
    for index in (0..source_rgdb_pyramid.octaves.len()).rev() {
        estimation_result = estimate(&source_rgdb_pyramid.octaves[index],depth_image,&target_rgdb_pyramid.octaves[index],index,&estimation_result,pinhole_camera,runtime_parameters);
    }


    lie::exp(&estimation_result.fixed_slice::<U3,U1>(0,0),&estimation_result.fixed_slice::<U3,U1>(0,0))
}

pub fn estimate(source_octave: &RGBDOctave, source_depth_image_original: &Image, target_octave: &RGBDOctave, octave_index: usize, initial_guess: &Vector6<Float>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> Vector6<Float> {
    let gray_image = &source_octave.gray_images[0];
    let x_gradients = &source_octave.x_gradients[0];
    let y_gradients = &source_octave.y_gradients[0];


    let identity = DMatrix::<Float>::identity(gray_image.buffer.nrows(),gray_image.buffer.ncols());

    panic!("estimate not implemented yet");
}

fn image_to_linear_index(r: usize, cols: usize, c: usize) -> usize {
    r*cols+c
}

fn linear_to_image_index(idx: usize, cols: usize) -> Point<usize> {
    let x = (idx as Float % cols as Float).trunc() as usize;
    let y =  ((idx  as Float) / (cols as Float)).trunc() as usize;
    Point::<usize>::new(x ,y)
}


fn image_gradients_as_matrix(x_gradients: &DMatrix<Float>, y_gradients: &DMatrix<Float>, target: &mut Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>) -> () {
    let (rows,cols) = x_gradients.shape();

    for r in 0..rows {
        for c in 0..cols {
            let linear_index = image_to_linear_index(r,cols,c);
            let x_grad = x_gradients[(r,c)];
            let y_grad = y_gradients[(r,c)];
            target[(linear_index,0)] = x_grad;
            target[(linear_index,1)] = y_grad;
        }
    }

}

fn backproject_points(source_image_buffer: &DMatrix<Float>,depth_image_buffer: &DMatrix<Float>,  pinhole_camera: &Pinhole, octave_index: usize) -> Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>> {
    let (rows,cols) = source_image_buffer.shape();
    let mut backproject_points =  Matrix::<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>::zeros(rows*cols);
    for r in 0..rows {
        for c in 0..cols {
            let depth_sample = depth_image_buffer[reconstruct_original_coordiantes(c, r, octave_index as u32)];
            let backprojected_point = pinhole_camera.backproject(&Point::<Float>::new(c as Float,r as Float), depth_sample);
            backproject_points.set_column(image_to_linear_index(r,cols,c),&Vector4::<Float>::new(backprojected_point[0],backprojected_point[1],backprojected_point[2],1.0));
        }
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

//TODO: Check for out of bounds
fn residual(target_image_buffer: &DMatrix<Float>,source_image_buffer: &DMatrix<Float>,backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>, est_transform: &Matrix4<Float>, pinhole_camera: &Pinhole, residual_target: &mut DVector<Float>) -> () {
    let transformed_points = est_transform*backprojected_points;
    let number_of_points = transformed_points.ncols();
    let image_width = target_image_buffer.ncols();

    for i in 0..number_of_points {
        let source_image_index = linear_to_image_index(i,image_width);
        let source_sample = source_image_buffer[(source_image_index.y,source_image_index.x)];

        let target_point = pinhole_camera.project(&transformed_points.fixed_slice::<U3,U1>(0,i));
        let target_sample = target_image_buffer[(target_point.y.trunc() as usize,target_point.x.trunc() as usize)];
        residual_target[i] = target_sample - source_sample;
    }

    panic!("Not finished")
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

fn full_jacobian(image_gradients: &Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>, const_jacobians: &Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>, target: &mut Matrix<Float,Dynamic,U6, VecStorage<Float,Dynamic,U6>>) -> () {
    let number_of_elements = image_gradients.nrows();

    for i in 0..number_of_elements {
        let jacobian_i = image_gradients.row(i)*const_jacobians.fixed_slice::<U2,U6>(i*2,0);
        target.copy_from(&jacobian_i);
    }

}

