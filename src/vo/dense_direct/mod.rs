extern crate nalgebra as na;

use na::{U1,U2,U3,U6,Vector2,Vector4,Vector6,DVector,Matrix4,Matrix,Matrix1x2,Matrix2x3,Matrix3x6,Matrix1x6,DMatrix,Dynamic,VecStorage};

use crate::pyramid::rgbd::{RGBDPyramid,rgbd_octave::RGBDOctave};
use crate::camera::{Camera,pinhole::Pinhole};
use crate::vo::dense_direct::dense_direct_runtime_parameters::DenseDirectRuntimeParameters;
use crate::image::Image;
use crate::numerics::lie::exp;
use crate::features::geometry::point::Point;
use crate::Float;

pub mod dense_direct_runtime_parameters;

pub fn run(source_rgdb_pyramid: &RGBDPyramid<RGBDOctave>,target_rgdb_pyramid: &RGBDPyramid<RGBDOctave>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> Matrix4<Float> {


    let depth_image = &source_rgdb_pyramid.depth_image;

    let mut estimation_result = Vector6::<Float>::zeros();
    
    for level in (0..source_rgdb_pyramid.octaves.len()).rev() {
        estimation_result = estimate(&source_rgdb_pyramid.octaves[level],depth_image,&target_rgdb_pyramid.octaves[level],level,&estimation_result,pinhole_camera,runtime_parameters);
    }


    exp(&estimation_result.fixed_slice::<U3,U1>(0,0),&estimation_result.fixed_slice::<U3,U1>(0,0))
}

pub fn estimate(source_octave: &RGBDOctave, source_depth_image_original: &Image, target_octave: &RGBDOctave, octave_level: usize, initial_guess: &Vector6<Float>, pinhole_camera: &Pinhole, runtime_parameters: &DenseDirectRuntimeParameters) -> Vector6<Float> {
    let gray_image = &source_octave.gray_images[0];
    let x_gradients = &source_octave.x_gradients[0];
    let y_gradients = &source_octave.y_gradients[0];



    let identity = DMatrix::<Float>::identity(gray_image.buffer.nrows(),gray_image.buffer.ncols());

    panic!("estimate not implemented yet");
}

fn linear_index(r: usize, cols: usize, c: usize) -> usize {
    r*cols+c
}


fn image_gradients_as_matrix(x_gradients: &DMatrix<Float>, y_gradients: &DMatrix<Float>, target: &mut Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>) -> () {
    let (rows,cols) = x_gradients.shape();

    for r in 0..rows {
        for c in 0..cols {
            let linear_index = linear_index(r,cols,c);
            let x_grad = x_gradients[(r,c)];
            let y_grad = y_gradients[(r,c)];
            target[(linear_index,0)] = x_grad;
            target[(linear_index,1)] = y_grad;
        }
    }

}

fn residual(source_image_buffer: &DMatrix<Float>, depth_image_buffer: &DMatrix<Float>,target_image_buffer: &DMatrix<Float>, pinhole_camera: &Pinhole, est_transform: &Matrix4<Float>, residual_target: &mut DVector<Float>) -> () {
    let (rows,cols) = source_image_buffer.shape();

    for r in 0..rows {
        for c in 0..cols {
            let depth = depth_image_buffer[(r,c)];
            let source_sample = source_image_buffer[(r,c)];
            let unprojected_pixel = pinhole_camera.unproject(&Point::new(c as Float, r as Float), depth);
            let transformed_point = est_transform*Vector4::<Float>::new(unprojected_pixel[0],unprojected_pixel[1],unprojected_pixel[2], 1.0);
            let target_point = pinhole_camera.project(&transformed_point.fixed_rows::<U3>(0));
            let target_sample = target_image_buffer[(target_point.y.trunc() as usize,target_point.x.trunc() as usize)];

            let linear_index = linear_index(r,cols,c);
            residual_target[linear_index] = target_sample - source_sample;
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

fn motion_jacobian(image_gradients: &Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>, camera_jacobian: &Matrix2x3<Float>, motion_jacobian: &Matrix3x6<Float>, target: &mut Matrix<Float,Dynamic,U6,VecStorage<Float,Dynamic,U6>>) -> () {
    target.copy_from(&(image_gradients*camera_jacobian*motion_jacobian));
}