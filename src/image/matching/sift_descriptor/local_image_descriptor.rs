extern crate nalgebra as na;

use na::{core::SMatrix,Vector2, Matrix2x4};
use crate::{Float,float};
use crate::numerics::{rotation_matrix_2d_from_orientation,gradient_and_orientation,gauss_2d};
use crate::image::pyramid::sift::sift_octave::SiftOctave;
use crate::image::matching::sift_descriptor::{ORIENTATION_BINS,DESCRIPTOR_BINS,orientation_histogram::OrientationHistogram,keypoint::KeyPoint};

const SUBMATRIX_LENGTH: usize = 4;
const SAMPLE_LENGTH: Float = 16.0;

#[derive(Debug,Clone)]
pub struct LocalImageDescriptor {
    pub x: usize,
    pub y: usize,
    pub descriptor_vector: Vec<OrientationHistogram>
}


//TODO: fix this
impl LocalImageDescriptor {
    pub fn new(octave: &SiftOctave,  keypoint: &KeyPoint) -> LocalImageDescriptor {
        let weighted_gradient_orientation_samples = generate_normalized_weight_orientation_arrays(&octave, keypoint);

        let weighted_sample_gradients = &weighted_gradient_orientation_samples.0;
        let sample_orientation = &weighted_gradient_orientation_samples.1;
        let mut descriptor = vec![OrientationHistogram::new(ORIENTATION_BINS);DESCRIPTOR_BINS];

        for i in 0..descriptor.len() {
            let column_histogram = i % SUBMATRIX_LENGTH;
            let row_histogram =  i / SUBMATRIX_LENGTH;

            let column_submatrix = column_histogram * SUBMATRIX_LENGTH;
            let row_submatrix = row_histogram*SUBMATRIX_LENGTH;

            let weighted_sample_gradient_slice = weighted_sample_gradients.fixed_slice::<4,4>(row_submatrix,column_submatrix);
            let sample_orientations_slice = sample_orientation.fixed_slice::<4,4>(row_submatrix,column_submatrix);
            for r in 0..weighted_sample_gradient_slice.nrows() {
                for c in 0..weighted_sample_gradient_slice.ncols() {
                    let weighted_gradient_orientation = (weighted_sample_gradient_slice[(r,c)],sample_orientations_slice[(r,c)]);

                    let sample_x = column_submatrix*SUBMATRIX_LENGTH + c;
                    let sample_y = row_submatrix*SUBMATRIX_LENGTH + r;

                    let closest_histograms = closest_histograms(SUBMATRIX_LENGTH as isize,column_histogram as isize,row_histogram as isize,c as isize,r as isize);

                    let (dist_x,dist_y) = get_normalized_distance_to_center_for_histogram(sample_x as Float, sample_y as Float, row_histogram as Float, column_histogram as Float, SUBMATRIX_LENGTH as Float, SAMPLE_LENGTH);
                    let weight = (1.0-dist_x)*(1.0-dist_y);
                    descriptor[i].add_measurement_to_adjecent_with_interp(weighted_gradient_orientation, keypoint.orientation,weight); 

                    for (r,c,square_distance) in closest_histograms {
                        if square_distance < std::usize::MAX {
                            let j = SUBMATRIX_LENGTH*r+c;
                            let (other_dist_x,other_dist_y) = get_normalized_distance_to_center_for_histogram(sample_x as Float, sample_y as Float, r as Float, c as Float, SUBMATRIX_LENGTH as Float, SAMPLE_LENGTH);
                            let weight = (1.0-other_dist_x)*(1.0-other_dist_y);
                            descriptor[j].add_measurement_to_adjecent_with_interp(weighted_gradient_orientation, keypoint.orientation,weight); 
                        }
                    }
                }
            }
        }

        LocalImageDescriptor{x: keypoint.x, y: keypoint.y, descriptor_vector: descriptor}
    }
}

pub fn is_rotated_keypoint_within_image(octave: &SiftOctave, keypoint: &KeyPoint) -> bool {
    let total_desciptor_side_length = 8.0; //TODO but this into input parameters or make this constant
    let image = &octave.images[keypoint.sigma_level];
    let keypoint_sigma = &octave.sigmas[keypoint.sigma_level];
    let rot_mat = rotation_matrix_2d_from_orientation(keypoint.orientation);
    let key_x = keypoint.x as Float;
    let key_y = keypoint.y as Float;
    //let inter_pixel_distance = Octave::inter_pixel_distance(keypoint.octave_level);
    let corner_coordinates = Matrix2x4::new(
        -total_desciptor_side_length,total_desciptor_side_length,-total_desciptor_side_length,total_desciptor_side_length,
        -total_desciptor_side_length,-total_desciptor_side_length,total_desciptor_side_length,total_desciptor_side_length);

    let rotated_corners = rot_mat*corner_coordinates;
    let mut valid = true;

    for i in 0..4 {
        let coordiantes = rotated_corners.fixed_slice::<2,1>(0,i);
        //TODO: refactor this into a method
        let x_norm = coordiantes[(0,0)]/(keypoint_sigma);
        let y_norm = coordiantes[(1,0)]/(keypoint_sigma);
        let x =  key_x + x_norm;
        let y =  key_y + y_norm;
        valid &= x >= 0.0 && x < image.buffer.ncols() as Float && y >= 0.0 && y < image.buffer.nrows() as Float && 
        x_norm <= total_desciptor_side_length/2.0 && x_norm >= -total_desciptor_side_length/2.0 && y_norm <= total_desciptor_side_length/2.0 && y_norm >= -total_desciptor_side_length/2.0;
    }

    valid

}

//TODO: Fix this
fn generate_normalized_weight_orientation_arrays(octave: &SiftOctave, keypoint: &KeyPoint) -> (SMatrix<Float, 16,16>,SMatrix<Float, 16,16>) {

    let total_desciptor_side_length = 8 as isize;
    //let submatrix_length = (total_desciptor_side_length/2) as isize;
    let sigma = 1.5*octave.sigmas[keypoint.sigma_level];
    //let lambda_descriptor = 6.0; //TODO: get this from runtime parms i.e. blur_half_factor
    let x_gradient = &octave.x_gradient[keypoint.sigma_level];
    let y_gradient = &octave.y_gradient[keypoint.sigma_level];
    let inter_pixel_distance = SiftOctave::inter_pixel_distance(keypoint.octave_level);

    let x_center = keypoint.x as Float;
    let y_center = keypoint.y as Float;
    let rot_mat = rotation_matrix_2d_from_orientation(keypoint.orientation);

    let mut sample_weights = SMatrix::<Float,16,16>::zeros();
    let mut sample_orientations = SMatrix::<Float,16,16>::zeros();

    for x_off in -total_desciptor_side_length..total_desciptor_side_length {
        for y_off in -total_desciptor_side_length..total_desciptor_side_length {
            let matrix_index = ((y_off+total_desciptor_side_length) as usize,(x_off+total_desciptor_side_length) as usize);

            let coordinates_vector = Vector2::new(sigma*(x_off as Float)/inter_pixel_distance,sigma*(y_off as Float)/inter_pixel_distance);
            let rotated_coordinates = rot_mat*coordinates_vector;

            let rot_x = x_center + rotated_coordinates[(0,0)]; 
            let rot_y = y_center + rotated_coordinates[(1,0)];  
            let gauss_weight = gauss_2d(x_center, y_center,rot_x, rot_y,sigma);
            let grad_orientation = gradient_and_orientation(x_gradient,y_gradient,rot_x.trunc() as usize,rot_y.trunc() as usize); 
            let weighted_gradient = grad_orientation.0*gauss_weight; 
            
            sample_weights[matrix_index] = weighted_gradient;
            sample_orientations[matrix_index] = (grad_orientation.1 - keypoint.orientation) % (2.0*float::consts::PI);

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








