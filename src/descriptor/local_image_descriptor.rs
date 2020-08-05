extern crate nalgebra as na;

use na::{Matrix,Dim,DimName,DefaultAllocator,MatrixN};
use na::allocator::Allocator;
use crate::{float,Float, ExtremaParameters, KeyPoint};
use crate::image::Image;
use crate::pyramid::octave::Octave;
use crate::descriptor::gauss_2d;

#[derive(Debug,Clone)]
pub struct LocalImageDescriptor {
    max_bin: usize,
    pub bin_range: Float,
    pub bins: Vec<Float>
}

fn generate_sample_array<S: Dim + DimName>(keypoint: &KeyPoint) -> MatrixN<Float, S> where DefaultAllocator: Allocator<Float, S, S>   {

    let side_length = S::try_to_usize().unwrap();
    let square_length = (side_length/2) as isize;
    let x_center = keypoint.x as Float;
    let y_center = keypoint.y as Float;

    let mut m = MatrixN::<Float,S>::zeros();

    for r in 0..side_length {
        for c in 0..side_length {
            let matrix_index = (r,c);
            let gauss_sample_offset_x = c as isize - square_length;
            let gauss_sample_offset_y = -(r as isize) + square_length;
            let x = x_center + gauss_sample_offset_x as Float;
            let y = y_center + gauss_sample_offset_y as Float;
            let sigma = 1.0; //TODO: correct value
            let gauss_weight = gauss_2d(x_center, y_center,x, y, sigma);
            
            m[matrix_index] = 1.0;
            //TODO: set weighted value
        }
    }

    m
}






