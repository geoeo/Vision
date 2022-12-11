use nalgebra as na;
use nalgebra_sparse;

use nalgebra_sparse::{CooMatrix, CscMatrix};
use na::{Matrix3, MatrixXx3, Rotation3};
use rand::{thread_rng, Rng};

use core::panic;
use std::collections::HashMap;
use crate::Float;



/**
    Rotation Coordiante Descent Parra et al.
 */
pub fn rcd(indexed_relative_rotations: &Vec<Vec<((usize, usize), Matrix3<Float>)>>) -> Vec<Vec<((usize, usize), Matrix3<Float>)>> {
    let index_to_matrix_map = generate_path_indices_to_matrix_map(indexed_relative_rotations);
    let relative_rotations_csc = generate_relative_rotation_matrix(&index_to_matrix_map,indexed_relative_rotations);
    let absolute_rotations = generate_absolute_rotation_matrix(&index_to_matrix_map);
    let absolute_rotations_transpose = absolute_rotations.transpose();
    let number_of_absolute_rotations = index_to_matrix_map.len();

    let max_epoch = 100; //TODO: config
    for epoch in 0..max_epoch {
        for k in 0..number_of_absolute_rotations {
            let col_start = 3*k;
            let mut W = MatrixXx3::<Float>::zeros(col_start);
            let csc_slice: Vec<_> = relative_rotations_csc.triplet_iter().filter(|&(i, j, v)| j >= col_start && j < col_start+3).collect();
            for (i,j,v) in csc_slice {
                W[(i,col_start-j)] = *v;
            }

            let BW = &absolute_rotations*(&absolute_rotations_transpose*&W);
            let A = &W.transpose()*BW;


        }
    }


    


    panic!("TODO");
}

fn generate_path_indices_to_matrix_map(path_indices: &Vec<Vec<((usize, usize), Matrix3<Float>)>>) -> HashMap<usize,usize> {
    // assuming the first element of each vector is always the (same) root
    let number_of_rotations = path_indices.iter().map(|x| x.len()).sum::<usize>() + 1;
    let mut index_map = HashMap::<usize,usize>::with_capacity(number_of_rotations);
    let mut index_counter = 0;
    for v in path_indices {
        for ((i_s, i_f), _) in v {
            if !index_map.contains_key(i_s) {
                index_map.insert(*i_s, index_counter);
                index_counter += 1;
            }

            if !index_map.contains_key(i_f) {
                index_map.insert(*i_f, index_counter);
                index_counter += 1;
            }
        }
    }

    index_map
}

fn generate_relative_rotation_matrix(index_to_matrix_map: &HashMap<usize,usize>, indexed_relative_rotations: &Vec<Vec<((usize, usize), Matrix3<Float>)>>) -> CscMatrix<Float> {
    let number_of_views = index_to_matrix_map.len();
    let mut rotations_coo = CooMatrix::<Float>::zeros(3*number_of_views, 3*number_of_views);

    for v in indexed_relative_rotations {
        for ((i_s, i_f), rotation) in v {
            let idx_s = index_to_matrix_map.get(i_s).expect("RCD: Index s not present");
            let idx_f = index_to_matrix_map.get(i_f).expect("RCD: Index f not present");
            let rotation_transpose = rotation.transpose();
            // Symmetric Matrix of transpose R_ij
            rotations_coo.push_matrix(3*idx_s, 3*idx_f, &rotation_transpose);
            rotations_coo.push_matrix(3*idx_f, 3*idx_s, &rotation_transpose);

        }
    }
    CscMatrix::from(&rotations_coo)
}

/**
 * This will be initialized with random rotations as stated in the paper
 */
fn generate_absolute_rotation_matrix(index_to_matrix_map: &HashMap<usize,usize>) -> MatrixXx3<Float> {
    let number_of_views = index_to_matrix_map.len();
    let mut absolute_rotations = MatrixXx3::<Float>::zeros(3*number_of_views);
    let mut rng = thread_rng();

    for i in 0..number_of_views{
        let rot = rng.gen::<Rotation3<Float>>();
        absolute_rotations.fixed_rows_mut::<3>(i).copy_from(&rot.matrix());
    }
    absolute_rotations
}