extern crate nalgebra as na;

use na::{Matrix,Dim,DimName,DefaultAllocator,MatrixN,Vector2};
use na::allocator::Allocator;
use crate::{float,Float, ExtremaParameters, KeyPoint};
use crate::image::Image;
use crate::pyramid::octave::Octave;
use crate::descriptor::{rotation_matrix_2d_from_orientation,gauss_2d,gradient_and_orientation,orientation_histogram::OrientationHistogram};

#[derive(Debug,Clone)]
pub struct LocalImageDescriptor {
    pub descriptor_vector: Vec<OrientationHistogram>
}

impl LocalImageDescriptor {
    //TODO: maybe make sampe side length also a generic param
    //TODO: maybe just pass one weight matrix and calculate grad_oreintation here instead of generate_weighted..
    pub fn new<S: Dim + DimName>(gradient_orientation_samples:  &(MatrixN<Float, S>,MatrixN<Float, S>), orientation_bins: usize) -> LocalImageDescriptor  where DefaultAllocator: Allocator<Float, S, S> {
        let sample_side_length = 4;
        let sample_gradients = &gradient_orientation_samples.0;
        let sample_orientation = &gradient_orientation_samples.1;
        let descriptor_bins = (sample_gradients.nrows()/sample_side_length).pow(2);
        let mut descriptor = vec![OrientationHistogram::new(orientation_bins);descriptor_bins];

        for i in 0..descriptor.len() {
            let column_offset = i % sample_side_length;
            let row_offset = i / sample_side_length;
            let sample_gradient_slice = sample_gradients.slice((row_offset,column_offset),(sample_side_length,sample_side_length));
            let sample_orientations_slice = sample_orientation.slice((row_offset,column_offset),(sample_side_length,sample_side_length));
            for r in 0..sample_gradient_slice.nrows() {
                for c in 0..sample_gradient_slice.ncols() {
                    let gradient_orientation = (sample_gradient_slice[(r,c)],sample_orientations_slice[(r,c)]);
                    descriptor[i].add_measurement_to_all_with_interp(gradient_orientation); 
                }
            }
        }

        let other_descriptor = descriptor.clone();

        for i in 0..other_descriptor.len() {
            let target_histogram = &mut descriptor[i];
            for j in 0..other_descriptor.len() {
                if j == i {
                    continue;
                }
                let other_histogram = &other_descriptor[j];
                let weight = 0.0; //TODO Correct weight
                target_histogram.add_histogram(other_histogram, weight);

            }
        }

        LocalImageDescriptor{descriptor_vector: descriptor}
    }
}

fn generate_weighted_sample_array<S: Dim + DimName>(image:&Image,x_gradient: &Image, y_gradient: &Image, keypoint: &KeyPoint) -> (MatrixN<Float, S>,MatrixN<Float, S>) where DefaultAllocator: Allocator<Float, S, S>   {

    let side_length = S::try_to_usize().unwrap();
    let square_length = (side_length/2) as isize;
    let x_center = keypoint.x as Float;
    let y_center = keypoint.y as Float;
    let rot_mat = rotation_matrix_2d_from_orientation(keypoint.orientation);

    let mut sample_weights = MatrixN::<Float,S>::zeros();
    let mut sample_orientations = MatrixN::<Float,S>::zeros();

    for r in 0..side_length {
        for c in 0..side_length {
            let matrix_index = (r,c);
            //TODO: check this arithmatic
            let gauss_sample_offset_x = c as isize - square_length;
            let gauss_sample_offset_y = -(r as isize) + square_length;
            let x = x_center + gauss_sample_offset_x as Float;
            let y = y_center + gauss_sample_offset_y as Float;
            let coordinates_vector = Vector2::new(x,y);
            let rotated_coordinates = rot_mat*coordinates_vector;
            let rot_x = rotated_coordinates[(0,0)].floor() as usize; 
            let rot_y = rotated_coordinates[(1,0)].floor() as usize; 
            let sigma = square_length as Float;
            let gauss_weight = gauss_2d(x_center, y_center,x, y, sigma);
            let grad_orientation =gradient_and_orientation(x_gradient,y_gradient,rot_x,rot_y);
            let weighted_gradient = grad_orientation.0*gauss_weight; 
            
            sample_weights[matrix_index] = weighted_gradient;
            sample_orientations[matrix_index] = grad_orientation.1;

        }
    }

    (sample_weights,sample_orientations)
}






