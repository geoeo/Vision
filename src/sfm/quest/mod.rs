extern crate nalgebra as na;
extern crate nalgebra_lapack;

use na::{SMatrix, Matrix3, SVector, OMatrix, Dynamic, RowSVector, RowDVector, linalg::SVD, Quaternion, UnitQuaternion, Const};
use nalgebra_lapack::Eigen;
use rand::seq::SliceRandom;
use crate::sensors::camera::Camera;
use crate::sfm::{epipolar::Essential,tensor::{essential_matrix_from_motion,compute_fundamental,calc_sampson_distance_inliers_for_fundamental}};
use crate::image::features::{Feature,Match};
use crate::Float;

pub mod constraints;


pub fn quest_ransac<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, camera_one: &C, camera_two: &C, epipolar_thresh: Float, ransac_it: usize) -> Essential {
    let inverse_projection_one = camera_one.get_inverse_projection();
    let inverse_projection_two = camera_two.get_inverse_projection();
    let mut max_inlier_count = 0;
    let mut best_essential: Option<Essential> = None;
    let sample_size = matches.len();
    //let sample_size = 11;
    for _ in 0..ransac_it {
        let mut m1 = SMatrix::<Float,3,5>::zeros();
        let mut m2 = SMatrix::<Float,3,5>::zeros();
        let samples: Vec<_> = matches.choose_multiple(&mut rand::thread_rng(), sample_size).map(|x| x.clone()).collect();
        for i in 0..5 {
            let s = &samples[i];

            let f_1 = s.feature_one.get_camera_ray(&inverse_projection_one);
            let f_2 = s.feature_two.get_camera_ray(&inverse_projection_two);

            // let f_1 = s.feature_one.get_as_3d_point(-1.0);
            // let f_2 = s.feature_two.get_as_3d_point(-1.0);

            m1.column_mut(i).copy_from(&f_1);
            m2.column_mut(i).copy_from(&f_2);
        }
        let (essential, _, _) = quest(&m1,&m2);
        let f = compute_fundamental(&essential, &camera_one.get_inverse_projection(), &camera_two.get_inverse_projection());
        let inliers = calc_sampson_distance_inliers_for_fundamental(&f,&samples[5..].to_vec(),epipolar_thresh);
        if inliers > max_inlier_count {
            max_inlier_count = inliers;
            best_essential = Some(essential);
        }
    }

    println!("Best inliner count for essential matrix was {} out of {} samples. That is {} %.", max_inlier_count, matches.len(), ((max_inlier_count as Float) / ((sample_size - 5) as Float)) * 100.0);
    best_essential.expect("No essential matrix could be computer via RANSAC")
}


/**
 * m, n:    Homogeneous coordinates of 5 matched feature points in the first  
//          and second coordinate frames. Each column of m or n has the 
//          format [u, v, -1]^T, where x and y are coordinates of the  
//          feature point on the image plane. Thus, m and n are 3*5 matrices, 
//          with one entries in the 3rd row.
 */
#[allow(non_snake_case)]
pub fn quest(m1: &SMatrix<Float,3,5>, m2: &SMatrix<Float,3,5>) -> (Essential, SVector<Float,5>, SVector<Float,5>) {

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
    
    let idx_1 = SMatrix::<usize,20,4>::from_rows(&[ 
        RowSVector::<usize,4>::from_vec(vec![0,     1,     2,     3]),
        RowSVector::<usize,4>::from_vec(vec![1,     4,     5,     6]),
        RowSVector::<usize,4>::from_vec(vec![2,     5,     7,     8]),
        RowSVector::<usize,4>::from_vec(vec![3,     6,     8,    9]),
        RowSVector::<usize,4>::from_vec(vec![4,    10,    11,    12]),
        RowSVector::<usize,4>::from_vec(vec![5,    11,    13,    14]),
        RowSVector::<usize,4>::from_vec(vec![6,    12,    14,    15]),
        RowSVector::<usize,4>::from_vec(vec![7,    13,    16,    17]),
        RowSVector::<usize,4>::from_vec(vec![8,    14,    17,    18]),
        RowSVector::<usize,4>::from_vec(vec![9,    15,     18,    19]),
        RowSVector::<usize,4>::from_vec(vec![10,    20,    21,    22]),
        RowSVector::<usize,4>::from_vec(vec![11,    21,    23,    24]),
        RowSVector::<usize,4>::from_vec(vec![12,    22,    24,    25]),
        RowSVector::<usize,4>::from_vec(vec![13,    23,    26,    27]),
        RowSVector::<usize,4>::from_vec(vec![14,    24,    27,    28]),
        RowSVector::<usize,4>::from_vec(vec![15,    25,    28,    29]),
        RowSVector::<usize,4>::from_vec(vec![16,    26,    30,    31]),
        RowSVector::<usize,4>::from_vec(vec![17,    27,    31,    32]),
        RowSVector::<usize,4>::from_vec(vec![18,    28,    32,    33]),
        RowSVector::<usize,4>::from_vec(vec![19,    29,    33,    34])]);

    // First column of Idx2 shows the row index of matrix B. The second,
    // third, and fourth columns indicate the row index of Bbar which should be
    // used respectively xV, yV, and zV.
    let idx_2 = SMatrix::<usize,15,4>::from_rows(&[ 
        RowSVector::<usize,4>::from_vec(vec![20,     0,     1,     2]),
        RowSVector::<usize,4>::from_vec(vec![21,     1,     3,     4]),
        RowSVector::<usize,4>::from_vec(vec![22,     2,     4,     5]),
        RowSVector::<usize,4>::from_vec(vec![23,     3,     6,     7]),
        RowSVector::<usize,4>::from_vec(vec![24,     4,     7,     8]),
        RowSVector::<usize,4>::from_vec(vec![25,     5,     8,    9]),
        RowSVector::<usize,4>::from_vec(vec![26,     6,    10,    11]),
        RowSVector::<usize,4>::from_vec(vec![27,     7,    11,    12]),
        RowSVector::<usize,4>::from_vec(vec![28,     8,    12,    13]),
        RowSVector::<usize,4>::from_vec(vec![29,     9,    13,    14]),
        RowSVector::<usize,4>::from_vec(vec![30,    10,    15,    16]),
        RowSVector::<usize,4>::from_vec(vec![31,    11,    16,    17]),
        RowSVector::<usize,4>::from_vec(vec![32,    12,    17,    18]),
        RowSVector::<usize,4>::from_vec(vec![33,    13,    18,    19]),
        RowSVector::<usize,4>::from_vec(vec![34,    14,    19,    20])]);  

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

    let svd_a = SVD::new(-A2,true,true);
    let b_bar = svd_a.solve(&A1,1e-6).expect("SVD Solve Failed in Quest");

    // Let 
    // V = [w^4, w^3*x, w^3*y, w^3*z, w^2*x^2, w^2*x*y, w^2*x*z, w^2*y^2, w^2*y*z, w^2*z^2, w*x^3, w*x^2*y, w*x^2*z, w*x*y^2, w*x*y*z, w*x*z^2, w*y^3, w*y^2*z, w*y*z^2, w*z^3, x^4, x^3*y, x^3*z, x^2*y^2, x^2*y*z, x^2*z^2, x*y^3, x*y^2*z, x*y*z^2, x*z^3, y^4, y^3*z, y^2*z^2, y*z^3, z^4]^T
    // then we have
    // x V = w Bx V   ,   y V = w By V   ,   z V = w Bz V
    for i in 0..20{
        b_x[(idx_1[(i,0)], idx_1[(i,1)])] = 1.0;
    }

    for i in 0..idx_2.nrows() {
        b_x.row_mut(idx_2[(i,0)]).copy_from(&b_bar.row(idx_2[(i,1)]));
    }

    
    let eigen = Eigen::new(b_x,false,true).expect("QuEST: Eigen Decomp Failed!");
    let (_,_,real_eigenvectors_option) = eigen.get_real_elements();
    let real_eigenvectors = real_eigenvectors_option.expect("QuEST: Extracting Right Eigenvectors Failed!");
    let mut V = OMatrix::<Float,Const<35>,Dynamic>::from_columns(&real_eigenvectors);
    // Correct the sign of each column s.t. the first element (i.e., w) is always positive
    for i in 0..V.ncols() {
        if V[(0,i)] < 0.0{
            V.column_mut(i).scale_mut(-1.0);
        }
    }

    // Recover quaternion elements  
    let w  = RowDVector::<Float>::from_iterator(V.ncols(), V.row(0).into_owned().iter().map(|&v| Float::powf(v,0.25)));
    let w3 = RowDVector::<Float>::from_iterator(V.ncols(), w.iter().map(|&v| Float::powf(v,3.0)));
    let x = V.row(1).into_owned().component_div(&w3);
    let y = V.row(2).component_div(&w3);
    let z = V.row(3).component_div(&w3);

    // Each column represents a candidate rotation
    let mut Q = OMatrix::<Float,Const<4>,Dynamic>::from_rows(&[w,x,y,z]);

    // Normalize s.t. each column of Q has norm 1
    for mut c in Q.column_iter_mut() {
        c /= c.apply_norm(&na::EuclideanNorm);
    }

    let (candidate_translations, candidate_depth_1, candidate_depth_2) = recover_translation_and_depth(m1,m2,&Q);
    cheirality_check(&Q, &candidate_translations, &candidate_depth_1, &candidate_depth_2)
}

#[allow(non_snake_case)]
fn recover_translation_and_depth(m1: &SMatrix<Float,3,5>, m2: &SMatrix<Float,3,5>, Q: &OMatrix<Float,Const<4>,Dynamic>) ->(OMatrix<Float,Const<3>,Dynamic>, OMatrix<Float,Const<5>,Dynamic>, OMatrix<Float,Const<5>,Dynamic>) {

    let n = Q.ncols();
    let mut T = OMatrix::<Float,Const<3>,Dynamic>::zeros(n);
    let mut Z1 = OMatrix::<Float,Const<5>,Dynamic>::zeros(n);
    let mut Z2 = OMatrix::<Float,Const<5>,Dynamic>::zeros(n);
    let I = Matrix3::<Float>::identity();

    for k in 0..n{
        let quat = UnitQuaternion::from_quaternion(Quaternion::<Float>::new(Q[(0,k)],Q[(1,k)],Q[(2,k)],Q[(3,k)]));
        let R = quat.to_rotation_matrix().matrix().clone();

        // Stack rigid motion constraints into matrix-vector form C * Y = 0
        let mut C = SMatrix::<Float,15,13>::zeros();
        for i in 0..5 {
            C.fixed_slice_mut::<3,3>(i*3,0).copy_from(&I);
            for j in i*2+3..i*2+4{
                C.fixed_slice_mut::<3,1>(i*3,j).copy_from(&(R*m1.column(i)));
                C.fixed_slice_mut::<3,1>(i*3,j+1).copy_from(&-m2.column(i));
            }
        }

        // The right singular vector corresponding to the zero singular value of C.
        let Y = nalgebra_lapack::SVD::new(C).expect("Five Point: SVD failed on A!").vt.row(12).transpose();

        let t = Y.fixed_rows::<3>(0);  // Translation vector
        let z = Y.fixed_rows::<10>(3).into_owned();  // Depths in both camera frames


        let z1 =  SVector::<Float,5>::new(z[0],z[2],z[4],z[6],z[8]); // Depths in camera frame 1
        let z2 =  SVector::<Float,5>::new(z[1],z[3],z[5],z[7],z[9]); // Depths in camera frame 2

        // Store the results
        T.column_mut(k).copy_from(&t);
        Z1.column_mut(k).copy_from(&z1);
        Z2.column_mut(k).copy_from(&z2);
    }

    (T,Z1,Z2)

}

#[allow(non_snake_case)]
fn cheirality_check(Q: &OMatrix<Float,Const<4>,Dynamic>, T: &OMatrix<Float,Const<3>,Dynamic>,  Z1: &OMatrix<Float,Const<5>,Dynamic>, Z2: &OMatrix<Float,Const<5>,Dynamic>) -> (Essential, SVector<Float,5>, SVector<Float,5>) {

    let mut best_essential = Essential::zeros();
    let mut best_depths_1 = SVector::<Float,5>::zeros();
    let mut best_depths_2 = SVector::<Float,5>::zeros();
    let mut best_det = Float::MAX;
    let mut best_depth_count = 0;

    for k in 0..Q.ncols(){
        let quat = UnitQuaternion::from_quaternion(Quaternion::<Float>::new(Q[(0,k)],Q[(1,k)],Q[(2,k)],Q[(3,k)]));
        let R = quat.to_rotation_matrix().matrix().clone();
        let t = T.column(k).into_owned();
        let depths_1 = Z1.column(k);
        let depths_2 = Z2.column(k);

        let essential = essential_matrix_from_motion(&t,&R);
        let det = essential.determinant().abs();

        let negative_depth_count_1 = depths_1.iter().fold(0, |acc, &v| {
            match v < 0.0 {
                true => acc + 1,
                false => acc
            }
        });

        let negative_depth_count_2 = depths_2.iter().fold(0, |acc, &v| {
            match v < 0.0 {
                true => acc + 1,
                false => acc
            }
        });

        let neg_depth_count = negative_depth_count_1 + negative_depth_count_2;
        if neg_depth_count > best_depth_count || ((neg_depth_count == best_depth_count) && (det < best_det)) {
            best_essential = essential;
            best_depths_1 = depths_1.into_owned();
            best_depths_2 = depths_2.into_owned();
            best_det = det;
            best_depth_count = neg_depth_count;
        }
    }

    (best_essential, best_depths_1, best_depths_2)

}