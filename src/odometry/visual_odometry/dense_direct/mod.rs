extern crate nalgebra as na;

use na::{U2,U4,Dim,DimName,Const,RowVector2,Vector4,DVector,Matrix4,Matrix,DMatrix,Dynamic,VecStorage};

use crate::pyramid::gd::{GDPyramid,gd_octave::GDOctave};
use crate::sensors::camera::Camera;
use crate::numerics::{lie,loss::LossFunction, weighting::WeightingFunction};
use crate::features::geometry::point::Point;
use crate::{Float,float,reconstruct_original_coordiantes_for_float};

pub mod solver;

pub struct RuntimeMemory<const C: usize> {
    pub weights_vec: DVector::<Float>,
    pub residuals: DVector::<Float>,
    pub new_residuals: DVector::<Float>,
    pub full_jacobian: Matrix::<Float, Dynamic, Const<C>, VecStorage<Float, Dynamic, Const<C>>>,
    pub image_gradients: Matrix::<Float, Dynamic, U2, VecStorage<Float, Dynamic, U2>>,
    pub image_gradient_points: Vec::<Point<usize>>,
    pub new_image_gradient_points: Vec::<Point<usize>>,
    pub rescaled_jacobian_target: Matrix::<Float, Dynamic, Const<C>, VecStorage<Float, Dynamic, Const<C>>>,
    pub rescaled_residual_target: DVector::<Float>

}

impl<const T: usize> RuntimeMemory<T> {

    pub fn new(size: usize) ->  RuntimeMemory<T>{

        RuntimeMemory{
            weights_vec: DVector::<Float>::zeros(size),
            residuals: DVector::<Float>::from_element(size, float::MAX),
            new_residuals: DVector::<Float>::from_element(size, float::MAX),
            full_jacobian:
                Matrix::<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>::zeros(
                    size,
                ),
            image_gradients:
                Matrix::<Float, Dynamic, U2, VecStorage<Float, Dynamic, U2>>::zeros(size),
            image_gradient_points: Vec::<Point<usize>>::with_capacity(size),
            new_image_gradient_points: Vec::<Point<usize>>::with_capacity(size),
            rescaled_jacobian_target:
                Matrix::<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>::zeros(
                    size,
                ),
            rescaled_residual_target: DVector::<Float>::zeros(size)

        }

    }

    pub fn from_pyramid(pyramid: &GDPyramid<GDOctave>) -> Vec<RuntimeMemory<T>> {
        pyramid.octaves.iter().map(|octave| RuntimeMemory::new(octave.gray_images[0].size())).collect::<Vec<RuntimeMemory<T>>>()
    }
}


pub fn compute_t_dist_weights(residuals: &DVector<Float>, weights_vec: &mut DVector<Float>, n: Float, t_dist_nu: Float, max_it: usize, eps: Float) -> () {

    let variance = estimate_t_dist_variance(n,residuals,t_dist_nu,max_it,eps);
    for i in 0..residuals.len() {
        let res = residuals[i];
        weights_vec[i] = compute_t_dist_weight(res,variance,t_dist_nu).sqrt();
    }
    
}


pub fn compute_t_dist_weight(residual: Float, variance: Float, t_dist_nu: Float) -> Float{
    (t_dist_nu + 1.0) / (t_dist_nu + residual.powi(2)/variance)
}

pub fn estimate_t_dist_variance(n: Float, residuals: &DVector<Float>, t_dist_nu: Float, max_it: usize, eps: Float) -> Float {
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

fn image_to_linear_index(r: usize, cols: usize, c: usize) -> usize {
    r*cols+c
}

fn linear_to_image_index(idx: usize, cols: usize) -> Point<usize> {
    let x = (idx as Float % cols as Float).trunc() as usize;
    let y =  ((idx  as Float) / (cols as Float)).trunc() as usize;
    Point::<usize>::new(x ,y)
}

pub fn compute_image_gradients(x_gradients: &DMatrix<Float>, y_gradients: &DMatrix<Float>,image_gradient_points: &Vec<Point<usize>>, target: &mut Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>) -> () {
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

pub fn backproject_points<C : Camera>(source_image_buffer: &DMatrix<Float>,depth_image_buffer: &DMatrix<Float>,  camera: &C, pyramid_scale: Float,  octave_index: usize) -> (Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>, Vec<bool>) {
    let (rows,cols) = source_image_buffer.shape();
    let mut backproject_points =  Matrix::<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>::zeros(rows*cols);
    let mut backproject_points_flags =  vec![false;rows*cols];
    let number_of_points = rows*cols;

    for i in 0..number_of_points {
        let image_point =  linear_to_image_index(i, cols);
        let r = image_point.y;
        let c = image_point.x;
        let reconstruced_coordiantes = reconstruct_original_coordiantes_for_float(c as Float + 0.5, r as Float + 0.5,pyramid_scale , octave_index as i32);
        let depth_sample = depth_image_buffer[(reconstruced_coordiantes.1.trunc() as usize,reconstruced_coordiantes.0.trunc() as usize)];


        if depth_sample != 0.0 {
            let backprojected_point = camera.backproject(&Point::<Float>::new(c as Float + 0.5,r as Float + 0.5), depth_sample); //TODO: inverse depth
            backproject_points.set_column(image_to_linear_index(r,cols,c),&Vector4::<Float>::new(backprojected_point[0],backprojected_point[1],backprojected_point[2],1.0));
            backproject_points_flags[image_to_linear_index(r,cols,c)] = true;
        } else {
            backproject_points.set_column(image_to_linear_index(r,cols,c),&Vector4::<Float>::new(0.0,0.0,0.0,1.0));
        }
    }
    (backproject_points,backproject_points_flags)
}

pub fn precompute_jacobians<C: Camera, T: Dim + DimName>(backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>,backprojected_points_flags: &Vec<bool>, camera: &C) -> Matrix<Float,Dynamic,T, VecStorage<Float,Dynamic,T>> {
    let number_of_points = backprojected_points.ncols();
    let mut precomputed_jacobians = Matrix::<Float,Dynamic,T, VecStorage<Float,Dynamic,T>>::zeros(2*number_of_points);

    for i in 0..number_of_points {
        if backprojected_points_flags[i] {
            let point = backprojected_points.fixed_slice::<3,1>(0,i);
            let camera_jacobian = camera.get_jacobian_with_respect_to_position(&point);
            let lie_jacobian = lie::left_jacobian_around_identity(&point);
            precomputed_jacobians.fixed_slice_mut::<2,6>(i*2,0).copy_from(&(camera_jacobian*lie_jacobian));
        }

        

    }

    precomputed_jacobians
}

//TODO: performance offender
pub fn compute_full_jacobian<const D: usize>(image_gradients: &Matrix<Float,Dynamic,U2, VecStorage<Float,Dynamic,U2>>, const_jacobians: &Matrix<Float,Dynamic,Const<D>, VecStorage<Float,Dynamic,Const<D>>>, target: &mut Matrix<Float,Dynamic,Const<D>, VecStorage<Float,Dynamic,Const<D>>>) -> () {
    let number_of_elements = image_gradients.nrows();

    for i in 0..number_of_elements {
        let jacobian_i = image_gradients.row(i)*const_jacobians.fixed_slice::<2, D>(i*2,0);
        target.fixed_slice_mut::<1,D>(i,0).copy_from(&jacobian_i);

    }

}

//TODO: highest performance offender
pub fn compute_residuals<C>(target_image_buffer: &DMatrix<Float>,source_image_buffer: &DMatrix<Float>,backprojected_points: &Matrix<Float,U4,Dynamic,VecStorage<Float,U4,Dynamic>>,
    backprojected_points_flags: &Vec<bool>, est_transform: &Matrix4<Float>, camera: &C, residual_target: &mut DVector<Float>, image_gradient_points: &mut Vec<Point<usize>>) -> () where C: Camera {
    let transformed_points = est_transform*backprojected_points;
    let number_of_points = transformed_points.ncols();
    let image_width = target_image_buffer.ncols();

    let (rows,cols) = source_image_buffer.shape();
    for i in 0..number_of_points {
        let source_point = linear_to_image_index(i,image_width);
        let target_point = camera.project(&transformed_points.fixed_slice::<3,1>(0,i)); //TODO: inverse depth
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
            //residual_target[i] = 1.0;  // This give a CRT error
            //residual_target[i] = 1e-8; 
        }

    }


}

//TODO: naming
pub fn norm(
    residuals: &DVector<Float>,
    weight_function: &Box<dyn WeightingFunction>,
    weights_vec: &mut DVector<Float>,
) -> () {
    for i in 0..residuals.len() {
        let res = residuals[i];
        weights_vec[i] = weight_function.cost(res).sqrt();


    }
}

pub fn weight_residuals_sparse(
    residual_target: &mut DVector<Float>,
    weights_vec: &DVector<Float>,
) -> () {
    residual_target.component_mul_assign(weights_vec);
}

//TODO: optimize
//TODO: performance offender
pub fn weight_jacobian_sparse<const T: usize>(
    jacobian: &mut Matrix<Float, Dynamic, Const<T>, VecStorage<Float, Dynamic, Const<T>>>,
    weights_vec: &DVector<Float>,
) -> () {
    let size = weights_vec.len();
    for i in 0..size {
        let weighted_row = jacobian.row(i)*weights_vec[i];
        jacobian.row_mut(i).copy_from(&weighted_row);
    }
}




