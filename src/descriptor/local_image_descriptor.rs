extern crate nalgebra as na;

use na::{Dim,MatrixN,Vector2,base::dimension::{U16,U4}};
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
    pub fn new(gradient_orientation_samples:  &(MatrixN<Float, U16>,MatrixN<Float, U16>),  keypoint: &KeyPoint) -> LocalImageDescriptor {
        let descriptor_bins = 16;
        let sample_side_length = 4;
        let orientation_bins = 8;

        let sample_gradients = &gradient_orientation_samples.0;
        let sample_orientation = &gradient_orientation_samples.1;
        let mut descriptor = vec![OrientationHistogram::new(orientation_bins);descriptor_bins];

        for i in 0..descriptor.len() {
            let column_histogram = i % sample_side_length;
            let row_histogram =  i / sample_side_length;

            let column_submatrix = column_histogram * sample_side_length;
            let row_submatrix = row_histogram*sample_side_length;

            let sample_gradient_slice = sample_gradients.fixed_slice::<U4,U4>(row_submatrix,column_submatrix);
            let sample_orientations_slice = sample_orientation.fixed_slice::<U4,U4>(row_submatrix,column_submatrix);
            for r in 0..sample_gradient_slice.nrows() {
                for c in 0..sample_gradient_slice.ncols() {
                    let gradient_orientation = (sample_gradient_slice[(r,c)],sample_orientations_slice[(r,c)]);
                    descriptor[i].add_measurement_to_adjecent_with_interp(gradient_orientation, keypoint.orientation); 

                    //TODO: Pick up to 3 adjecent histrograms -  current one is already set
                    let window_x = column_submatrix*sample_side_length + c;
                    let window_y = row_submatrix*sample_side_length + r;

                    let closest_histograms = closest_histograms(sample_side_length as isize,column_histogram as isize,row_histogram as isize,c as isize,r as isize);


                }
            }
        }

        LocalImageDescriptor{descriptor_vector: descriptor}
    }
}

fn generate_weighted_sample_array(x_gradient: &Image, y_gradient: &Image, keypoint: &KeyPoint) -> (MatrixN<Float, U16>,MatrixN<Float, U16>) {

    let side_length = U16::try_to_usize().unwrap();
    let square_length = (side_length/2) as isize;
    let x_center = keypoint.x as Float;
    let y_center = keypoint.y as Float;
    let rot_mat = rotation_matrix_2d_from_orientation(keypoint.orientation);

    let mut sample_weights = MatrixN::<Float,U16>::zeros();
    let mut sample_orientations = MatrixN::<Float,U16>::zeros();

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

fn closest_histograms(side_length: isize, column_histogram: isize, row_histogram: isize, delta_c: isize, delta_r: isize) -> Vec::<(usize,usize,usize)> {

    let window_x = column_histogram*side_length + delta_c;
    let window_y = row_histogram*side_length + delta_r;

    
    let left_histogram = (row_histogram,column_histogram-1);
    let right_histogram = (row_histogram,column_histogram+1);
    let top_histogram = (row_histogram-1,column_histogram);
    let bottom_histogram = (row_histogram+1,column_histogram);
    let top_left_histogram = (row_histogram-1,column_histogram-1);
    let top_right_histogram = (row_histogram-1,column_histogram+1);
    let bottom_left_histogram = (row_histogram+1,column_histogram-1);
    let bottom_right_histogram = (row_histogram+1,column_histogram+1);

    let possible_histrograms 
        = vec![left_histogram,right_histogram,top_histogram,bottom_histogram,top_left_histogram,top_right_histogram,bottom_left_histogram,bottom_right_histogram];
    let valid_histograms = possible_histrograms.into_iter().filter(|&(r,c)| r >= 0 && c >= 0).collect::<Vec<_>>();
    let histogram_distances = valid_histograms.into_iter().map(|(r,c)| {
        let x = r*side_length;
        let y = c*side_length;

        let square_distance = (x-window_x).pow(2) + (y-window_y).pow(2);
        (r as usize,c as usize,square_distance as usize)

    } ).collect::<Vec<_>>();
    
    let mut closest_positions = vec![(std::usize::MAX,std::usize::MAX,std::usize::MAX);3];

    //TODO: maybe this can be done more elegantly
    for(r,c,square_distance) in histogram_distances {

        if square_distance < closest_positions[0].2 {
            closest_positions[2] = closest_positions[1];
            closest_positions[1] = closest_positions[0];
            closest_positions[0] = (r,c,square_distance);
        } 
        
        else if square_distance < closest_positions[1].2 {
            closest_positions[2] = closest_positions[1];
            closest_positions[1] = (r,c,square_distance);
        }  
        
        else if square_distance < closest_positions[2].2 {
            closest_positions[2] = (r,c,square_distance);
        }

    }

    closest_positions
}






