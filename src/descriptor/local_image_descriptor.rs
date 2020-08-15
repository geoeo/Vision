extern crate nalgebra as na;

use na::{Dim,MatrixN,Vector2,base::dimension::{U16,U4,U2,U1}, Matrix2x4};
use crate::{Float, KeyPoint};
use crate::image::Image;
use crate::pyramid::octave::Octave;
use crate::descriptor::{rotation_matrix_2d_from_orientation,gauss_2d,gradient_and_orientation,orientation_histogram::OrientationHistogram};

#[derive(Debug,Clone)]
pub struct LocalImageDescriptor {
    pub descriptor_vector: Vec<OrientationHistogram>
}

impl LocalImageDescriptor {
    pub fn new(octave: &Octave,  keypoint: &KeyPoint) -> LocalImageDescriptor {
        let descriptor_bins = 16;
        let submatrix_length = 4;
        let orientation_bins = 8;
        let sample_length = 16;

        let weighted_gradient_orientation_samples = generate_weighted_sample_array(&octave.x_gradient[keypoint.sigma_level], &octave.y_gradient[keypoint.sigma_level], keypoint);

        let weighted_sample_gradients = &weighted_gradient_orientation_samples.0;
        let sample_orientation = &weighted_gradient_orientation_samples.1;
        let mut descriptor = vec![OrientationHistogram::new(orientation_bins);descriptor_bins];

        for i in 0..descriptor.len() {
            let column_histogram = i % submatrix_length;
            let row_histogram =  i / submatrix_length;

            let column_submatrix = column_histogram * submatrix_length;
            let row_submatrix = row_histogram*submatrix_length;

            let weighted_sample_gradient_slice = weighted_sample_gradients.fixed_slice::<U4,U4>(row_submatrix,column_submatrix);
            let sample_orientations_slice = sample_orientation.fixed_slice::<U4,U4>(row_submatrix,column_submatrix);
            for r in 0..weighted_sample_gradient_slice.nrows() {
                for c in 0..weighted_sample_gradient_slice.ncols() {
                    let weighted_gradient_orientation = (weighted_sample_gradient_slice[(r,c)],sample_orientations_slice[(r,c)]);

                    let sample_x = column_submatrix*submatrix_length + c;
                    let sample_y = row_submatrix*submatrix_length + r;

                    let closest_histograms = closest_histograms(submatrix_length as isize,column_histogram as isize,row_histogram as isize,c as isize,r as isize);

                    let (dist_x,dist_y) = get_normalized_distance_to_center_for_histogram(sample_x as Float, sample_y as Float, row_histogram as Float, column_histogram as Float, submatrix_length as Float, sample_length as Float);
                    let weight = (1.0-dist_x)*(1.0-dist_y);
                    descriptor[i].add_measurement_to_adjecent_with_interp(weighted_gradient_orientation, keypoint.orientation,weight); 

                    for (r,c,square_distance) in closest_histograms {
                        if square_distance < std::usize::MAX {
                            let j = submatrix_length*r+c;
                            let (other_dist_x,other_dist_y) = get_normalized_distance_to_center_for_histogram(sample_x as Float, sample_y as Float, r as Float, c as Float, submatrix_length as Float, sample_length as Float);
                            let weight = (1.0-other_dist_x)*(1.0-other_dist_y);
                            descriptor[j].add_measurement_to_adjecent_with_interp(weighted_gradient_orientation, keypoint.orientation,weight); 
                        }
                    }
                }
            }
        }

        LocalImageDescriptor{descriptor_vector: descriptor}
    }
}

pub fn is_rotated_keypoint_within_image(octave: &Octave, keypoint: &KeyPoint) -> bool {
    let half_sample_length = 8.0;
    let image = &octave.images[keypoint.sigma_level];
    let rot_mat = rotation_matrix_2d_from_orientation(keypoint.orientation);
    let key_x = keypoint.x as Float;
    let key_y = keypoint.y as Float;
    let corner_coordinates = Matrix2x4::new(key_x-half_sample_length,key_x+half_sample_length,key_x-half_sample_length,key_x+half_sample_length,
        key_y-half_sample_length ,key_y-half_sample_length,key_y+half_sample_length,key_y+half_sample_length);

    let rotated_corners = rot_mat*corner_coordinates;
    let mut valid = true;

    for i in 0..4 {
        let coordiantes = rotated_corners.fixed_slice::<U2,U1>(0,i);
        let x =  coordiantes[(0,0)];
        let y =  coordiantes[(1,0)];
        valid &= x >= 0.0 && x < image.buffer.ncols() as Float && y >= 0.0 && y < image.buffer.nrows() as Float;
    }

    valid

}

fn generate_weighted_sample_array(x_gradient: &Image, y_gradient: &Image, keypoint: &KeyPoint) -> (MatrixN<Float, U16>,MatrixN<Float, U16>) {

    let side_length = U16::try_to_usize().unwrap();
    let submatrix_length = (side_length/2) as isize;
    let x_center = keypoint.x as Float;
    let y_center = keypoint.y as Float;
    let rot_mat = rotation_matrix_2d_from_orientation(keypoint.orientation);

    let mut sample_weights = MatrixN::<Float,U16>::zeros();
    let mut sample_orientations = MatrixN::<Float,U16>::zeros();

    for r in 0..side_length {
        for c in 0..side_length {
            let matrix_index = (r,c);
            let gauss_sample_offset_x = c as isize - submatrix_length;
            let gauss_sample_offset_y = -(r as isize) + submatrix_length;
            let x = x_center + gauss_sample_offset_x as Float;
            let y = y_center + gauss_sample_offset_y as Float;
            let coordinates_vector = Vector2::new(x,y);
            let rotated_coordinates = rot_mat*coordinates_vector;
            let rot_x = rotated_coordinates[(0,0)].floor() as usize; 
            let rot_y = rotated_coordinates[(1,0)].floor() as usize; 
            let sigma = submatrix_length as Float;
            let gauss_weight = gauss_2d(x_center, y_center,x, y, sigma);
            let grad_orientation = gradient_and_orientation(x_gradient,y_gradient,rot_x,rot_y); //TODO: this may go out of bounds due to rotation
            let weighted_gradient = grad_orientation.0*gauss_weight; 
            
            sample_weights[matrix_index] = weighted_gradient;
            sample_orientations[matrix_index] = grad_orientation.1;

        }
    }

    (sample_weights,sample_orientations)
}

fn closest_histograms(side_length: isize, column_histogram: isize, row_histogram: isize, delta_c: isize, delta_r: isize) -> Vec::<(usize,usize,usize)> {

    let sample_x = column_histogram*side_length + delta_c;
    let sample_y = row_histogram*side_length + delta_r;

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
    let valid_histograms = possible_histrograms.into_iter().filter(|&(r,c)| r >= 0 && c >= 0 && r < side_length && c < side_length).collect::<Vec<_>>();
    let histogram_distances = valid_histograms.into_iter().map(|(r,c)| {
        let x = r*side_length;
        let y = c*side_length;

        let square_distance = (x-sample_x).pow(2) + (y-sample_y).pow(2);
        (r as usize,c as usize,square_distance as usize)

    } ).collect::<Vec<_>>();
    
    let mut closest_positions = vec![(std::usize::MAX,std::usize::MAX,std::usize::MAX);3];

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

fn get_normalized_distance_to_center_for_histogram(sample_x: Float, sample_y: Float, histogram_row: Float, histogram_column: Float, submatrix_length: Float, sample_length: Float) -> (Float,Float) {

    let center = sample_length/2.0;
    let max_diff = sample_length/center;

    let submatrix_x_norm = (histogram_column*submatrix_length - center)/center;
    let submatrix_y_norm = (histogram_row*submatrix_length - center) / center;

    let sample_x_norm = (sample_x-center)/center;
    let sample_y_norm = (sample_y-center)/center;

    let delta_x = (submatrix_x_norm-sample_x_norm).abs();
    let delta_y = (submatrix_y_norm-sample_y_norm).abs();

    match (submatrix_x_norm,submatrix_y_norm) {
        (x,y) if x < 0.0 && y < 0.0 => (max_diff - delta_x,delta_y),
        (x,y) if x < 0.0 && y > 0.0 => (max_diff - delta_x, max_diff-delta_y),
        (x,y) if x > 0.0 && y < 0.0 => (delta_x,delta_y),
        (x,y) if x > 0.0 && y > 0.0 => (delta_x,max_diff-delta_y),
        (_,_) => (delta_x,delta_y)
    }

}






