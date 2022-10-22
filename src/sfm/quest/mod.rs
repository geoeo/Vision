extern crate nalgebra as na;

use na::{SMatrix, SVector, RowSVector, Isometry3};
use crate::Float;

pub mod constraints;



// % First column of Idx1 shows the row index of matrix B. The second, 
// % third, and fourth columns indicate the column index of B which should  
// % contain a 1 element respectively for xV, yV, and zV.
// % V = [w^4, w^3*x, w^3*y, w^3*z, w^2*x^2, w^2*x*y, w^2*x*z, w^2*y^2, w^2*y*z, w^2*z^2, w*x^3, w*x^2*y, w*x^2*z, w*x*y^2, w*x*y*z, w*x*z^2, w*y^3, w*y^2*z, w*y*z^2, w*z^3, x^4, x^3*y, x^3*z, x^2*y^2, x^2*y*z, x^2*z^2, x*y^3, x*y^2*z, x*y*z^2, x*z^3, y^4, y^3*z, y^2*z^2, y*z^3, z^4]^T
// Idx1 = [ 1     2     3     4
//          2     5     6     7
//          3     6     8     9
//          4     7     9    10
//          5    11    12    13
//          6    12    14    15
//          7    13    15    16
//          8    14    17    18
//          9    15    18    19
//         10    16    19    20
//         11    21    22    23
//         12    22    24    25
//         13    23    25    26
//         14    24    27    28
//         15    25    28    29
//         16    26    29    30
//         17    27    31    32
//         18    28    32    33
//         19    29    33    34
//         20    30    34    35];

// % First column of Idx2 shows the row index of matrix B. The second,
// % third, and fourth columns indicate the row index of Bbar which should be
// % used respectively xV, yV, and zV.
// Idx2 = [ 21     1     2     3
//          22     2     4     5
//          23     3     5     6
//          24     4     7     8
//          25     5     8     9
//          26     6     9    10
//          27     7    11    12
//          28     8    12    13
//          29     9    13    14
//          30    10    14    15
//          31    11    16    17
//          32    12    17    18
//          33    13    18    19
//          34    14    19    20
//          35    15    20    21];     

// Bx = zeros(35); 


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
        let idx_w = SVector::<usize,35>::from_iterator(0..35);
        //Index of the rest of the columns (monomials with no power of w)
        let idx_w0 = SVector::<usize,35>::from_iterator(35..56);


        panic!("TODO");

}