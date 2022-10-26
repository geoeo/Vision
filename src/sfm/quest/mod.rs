extern crate nalgebra as na;
extern crate nalgebra_lapack;

use na::{SMatrix, SVector, RowSVector, Isometry3, linalg::SVD};
use nalgebra_lapack::Eigen;
use crate::Float;

use num_complex::Complex;

pub mod constraints;


/**
 * m, n:    Homogeneous coordinates of 5 matched feature points in the first  
//          and second coordinate frames. Each column of m or n has the 
//          format [u, v, -1]^T, where x and y are coordinates of the  
//          feature point on the image plane. Thus, m and n are 3*5 matrices, 
//          with one entries in the 3rd row.
 */
pub fn quest(m1: &SMatrix<Float,3,5>, m2: &SMatrix<Float,3,5>) -> Isometry3<Float> {

    /*
        Let 
        V:   vector of all degree 4 monomials (with 35 entries)
        X:   vector of all degree 5 monomials (with 56 entries) in variables w,x,y,z. 
        Rows of Idx respectively show the index of the entries of vectors w*V, 
        x*V, y*V, and z*V, in the vector X.
    */
    let idx = SMatrix::<usize,4,35>::from_rows(&[
        RowSVector::<usize,35>::from_vec(vec![0,1,4,10,20,2,5,11,21,7,13,23,16,26,30,3,6,12,22,8,14,24,17,27,31,9,15,25,18,28,32,19,29,33,34]),
        RowSVector::<usize,35>::from_vec(vec![1,4,10,20,35,5,11,21,36,13,23,38,26,41,45,6,12,22,37,14,24,39,27,42,46,15,25,40,28,43,47,29,44,48,49]),
        RowSVector::<usize,35>::from_vec(vec![2,5,11,21,36,7,13,23,38,16,26,41,30,45,50,8,14,24,39,17,27,42,31,46,51,18,28,43,32,47,52,33,48,53,54]),
        RowSVector::<usize,35>::from_vec(vec![3,6,12,22,37,8,14,24,39,17,27,42,31,46,51,9,15,25,40,18,28,43,32,47,52,19,29,44,33,48,53,34,49,54,55])]);

    // Index of columns of A corresponding to all monomials with a least one power of w
    let idx_w_start = 0;
    //Index of the rest of the columns (monomials with no power of w)
    let idx_w0_start = 35;

    // First column of Idx1 shows the row index of matrix B. The second, 
    // third, and fourth columns indicate the column index of B which should  
    // contain a 1 element respectively for xV, yV, and zV.
    // V = [w^4, w^3*x, w^3*y, w^3*z, w^2*x^2, w^2*x*y, w^2*x*z, w^2*y^2, w^2*y*z, w^2*z^2, w*x^3, w*x^2*y, w*x^2*z, w*x*y^2, w*x*y*z, w*x*z^2, w*y^3, w*y^2*z, w*y*z^2, w*z^3, x^4, x^3*y, x^3*z, x^2*y^2, x^2*y*z, x^2*z^2, x*y^3, x*y^2*z, x*y*z^2, x*z^3, y^4, y^3*z, y^2*z^2, y*z^3, z^4]^T
    
    let idx_1 = SVector::<usize,100>::from_vec(vec![ 
             0,     1,     2,     3,
             1,     4,     5,     6,
             2,     5,     7,     8,
             3,     6,     8,    9,
             4,    10,    11,    12,
             5,    11,    13,    14,
             6,    12,    14,    15,
             7,    13,    16,    17,
             8,    14,    17,    18,
            9,    15,     18,    19,
            10,    20,    21,    22,
            11,    21,    23,    24,
            12,    22,    24,    25,
            13,    23,    26,    27,
            14,    24,    27,    28,
            15,    25,    28,    29,
            16,    26,    30,    31,
            17,    27,    31,    32,
            18,    28,    32,    33,
            19,    29,    33,    34]);

    // First column of Idx2 shows the row index of matrix B. The second,
    // third, and fourth columns indicate the row index of Bbar which should be
    // used respectively xV, yV, and zV.
    let idx_2 = SVector::<usize,60>::from_vec(vec![ 
                20,     0,     1,     2,
                21,     1,     3,     4,
                22,     2,     4,     5,
                23,     3,     6,     7,
                24,     4,     7,     8,
                25,     5,     8,    9,
                26,     6,    10,    11,
                27,     7,    11,    12,
                28,     8,    12,    13,
                29,     9,    13,    14,
                30,    10,    15,    16,
                31,    11,    16,    17,
                32,    12,    17,    18,
                33,    13,    18,    19,
                34,    14,    19,    20]);  

    let mut b_x = SMatrix::<Float,35,35>::zeros();      
    let c_f = constraints::generate_constraints(m1, m2);

    // A is the coefficient matrix such that A * X = 0
    let mut A = SMatrix::<Float,40,56>::zeros();
    for i in 0..4 {
        for j in 0..35 {
            for k in 0..10 {
                let j_prime = idx[(i,j)];
                A[(i*10+k,j_prime)] = c_f[(k,j)];
            }
            
        }

    }

    
    // Split A into matrices A1 and A2. A1 corresponds to terms that contain w, 
    // and A2 corresponds to the rest of the terms.
    let A1 = A.fixed_columns::<35>(idx_w_start).into_owned();
    let A2 = A.fixed_columns::<21>(idx_w0_start).into_owned();

    let svd_a = SVD::new(-A2,false,false);
    let b_bar = svd_a.solve(&A1,1e-6).expect("SVD Solve Failed in Quest");

    // Let 
    // V = [w^4, w^3*x, w^3*y, w^3*z, w^2*x^2, w^2*x*y, w^2*x*z, w^2*y^2, w^2*y*z, w^2*z^2, w*x^3, w*x^2*y, w*x^2*z, w*x*y^2, w*x*y*z, w*x*z^2, w*y^3, w*y^2*z, w*y*z^2, w*z^3, x^4, x^3*y, x^3*z, x^2*y^2, x^2*y*z, x^2*z^2, x*y^3, x*y^2*z, x*y*z^2, x*z^3, y^4, y^3*z, y^2*z^2, y*z^3, z^4]^T
    // then we have
    // x V = w Bx V   ,   y V = w By V   ,   z V = w Bz V
    for i in 0..20{
        b_x[(idx_1[(i,1)], idx_1[(i,2)])] = 1.0;
    }

    //Bx(Idx2(:,1),:) = Bbar(Idx2(:,2),:); TODO: check this
    for i in 0..idx_2.nrows() {
        b_x.row_mut(idx_2[(i,0)]).copy_from(&b_bar.row(idx_2[(i,1)]));
    }

    
    let eigen = Eigen::new(b_x,false,true).expect("QuEST: Eigen Decomp Failed!");
    let (_,_,real_eigenvectors_option) = eigen.get_real_elements();
    let real_eigenvectors = real_eigenvectors_option.expect("QuEST: Extracting Right Eigenvectors Failed!");
    let mut V = SMatrix::<Float,35,35>::from_columns(&real_eigenvectors);
    // Correct the sign of each column s.t. the first element (i.e., w) is always positive
    for i in 0..35 {
        if V[(0,i)] < 0.0{
            V[(0,i)] *= -1.0;
        }
    }


    // Recover quaternion elements  
    let w  = RowSVector::<Float,35>::from_iterator(V.row(0).into_owned().iter().map(|&v| Float::powf(v,0.25)));
    let w3 = RowSVector::<Float,35>::from_iterator(w.iter().map(|&v| Float::powf(v,3.0)));
    let x = V.row(1).into_owned().component_div(&w3);
    let y = V.row(2).component_div(&w3);
    let z = V.row(3).component_div(&w3);

    let mut Q = SMatrix::<Float,4,35>::from_rows(&[w,x,y,z]);

    // Normalize s.t. each column of Q has norm 1
    for mut c in Q.column_iter_mut() {
        c /= c.apply_norm(&na::EuclideanNorm);
    }

        
    // TODO: Translation and Depth
    panic!("TODO");
}