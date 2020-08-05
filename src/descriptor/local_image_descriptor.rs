extern crate nalgebra as na;

use na::{Matrix,Dim,DimName,DefaultAllocator,MatrixN,Vector2};
use na::allocator::Allocator;
use crate::{float,Float, ExtremaParameters, KeyPoint};
use crate::image::Image;
use crate::pyramid::octave::Octave;
use crate::descriptor::{rotation_matrix_2d_from_orientation,gauss_2d};

#[derive(Debug,Clone)]
pub struct LocalImageDescriptor {
    max_bin: usize,
    pub bin_range: Float,
    pub bins: Vec<Float>
}

fn generate_sample_array<S: Dim + DimName>(image:&Image, keypoint: &KeyPoint) -> MatrixN<Float, S> where DefaultAllocator: Allocator<Float, S, S>   {

    let side_length = S::try_to_usize().unwrap();
    let square_length = (side_length/2) as isize;
    let x_center = keypoint.x as Float;
    let y_center = keypoint.y as Float;
    let rot_mat = rotation_matrix_2d_from_orientation(keypoint.orientation);

    let mut m = MatrixN::<Float,S>::zeros();

    for r in 0..side_length {
        for c in 0..side_length {
            let matrix_index = (r,c);
            let gauss_sample_offset_x = c as isize - square_length;
            let gauss_sample_offset_y = -(r as isize) + square_length;
            let x = x_center + gauss_sample_offset_x as Float;
            let y = y_center + gauss_sample_offset_y as Float;
            let coordinates_vector = Vector2::new(x,y);
            let rotated_coordinates = rot_mat*coordinates_vector;
            let rot_x = rotated_coordinates[(0,0)].floor() as usize; 
            let rot_y = rotated_coordinates[(1,0)].floor() as usize; 
            let sigma = 1.0; //TODO: correct value
            let gauss_weight = gauss_2d(x_center, y_center,x, y, sigma);
            let weighted_value = image.buffer[(rot_y,rot_x)]*gauss_weight; //TODO: use correct weighting
            
            m[matrix_index] = weighted_value;

        }
    }

    m
}






