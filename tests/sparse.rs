use nalgebra as na;
use nalgebra_sparse;
use rand::{thread_rng, Rng};

use nalgebra_sparse::{CooMatrix, CscMatrix};
use na::{MatrixXx3,Rotation3};
use vision::sfm::rotation_avg::generate_dense_from_csc_slice;
use vision::Float;

#[test]
fn test_generate_dense_from_csc_slice() {
    let mut rng = thread_rng();    
    let rot = rng.gen::<Rotation3<Float>>();
    let rot_matrix = rot.matrix();
    let rot_matrix_transpose = rot_matrix.transpose();

    let number_of_views = 2;
    let mut rotations_coo = CooMatrix::<Float>::zeros(3*number_of_views, 3*number_of_views);
    rotations_coo.push_matrix(3, 0, &rot_matrix);
    rotations_coo.push_matrix(0, 3, &rot_matrix_transpose);

    let csc_matrix = CscMatrix::from(&rotations_coo);
    let dense_slice = generate_dense_from_csc_slice(0,2,&csc_matrix);

    let mut ground_truth_slice = MatrixXx3::<Float>::zeros(6);
    ground_truth_slice.fixed_slice_mut::<3,3>(3,0).copy_from(&rot_matrix);

    assert_eq!(ground_truth_slice,dense_slice);
}