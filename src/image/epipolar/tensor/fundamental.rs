extern crate nalgebra as na;
extern crate rand;

use na::{Matrix3,Matrix,Dynamic, VecStorage, dimension::U9};

use crate::Float;
use crate::image::features::{Feature,Match};
use crate::image::epipolar::tensor::Fundamental;

/**
 * Photogrammetric Computer Vision p.570
 * Fails if points are coplanar!
 */
#[allow(non_snake_case)]
pub fn eight_point<T : Feature>(matches: &Vec<Match<T>>, positive_principal_distance: bool) -> Fundamental {
    let number_of_matches = matches.len() as Float; 
    assert!(number_of_matches >= 8.0);


    let mut A = Matrix::<Float,Dynamic, U9, VecStorage<Float,Dynamic,U9>>::zeros(matches.len());
    for i in 0..A.nrows() {
        let feature_right = matches[i].feature_two.get_as_2d_point();
        let feature_left = matches[i].feature_one.get_as_2d_point();

        let l_x =  feature_left[0];
        let l_y =  feature_left[1];

        let r_x =  feature_right[0];
        let r_y =  feature_right[1];

        A[(i,0)] = r_x*l_x;
        A[(i,1)] = r_x*l_y;
        A[(i,2)] = r_x;
        A[(i,3)] = r_y*l_x;
        A[(i,4)] = r_y*l_y;
        A[(i,5)] = r_y;
        A[(i,6)] = l_x;
        A[(i,7)] = l_y;
        A[(i,8)] = 1.0;
    }


    let svd = A.svd(false,true);
    let v_t =  &svd.v_t.expect("SVD failed on A");
    let f = &v_t.row(v_t.nrows()-1);
    let mut F = Matrix3::<Float>::zeros();
    F[(0,0)] = f[0];
    F[(1,0)] = f[1];
    F[(2,0)] = f[2];
    F[(0,1)] = f[3];
    F[(1,1)] = f[4];
    F[(2,1)] = f[5];
    F[(0,2)] = f[6];
    F[(1,2)] = f[7];
    F[(2,2)] = f[8];

    let mut svd_f = F.svd(true,true);
    let acc = svd_f.singular_values[0].powi(2) + svd_f.singular_values[1].powi(2);
    svd_f.singular_values[2] = 0.0;
    svd_f.singular_values /= acc.sqrt();
    match positive_principal_distance {
        true => svd_f.recompose().ok().expect("SVD recomposition failed").transpose(),
        false => svd_f.recompose().ok().expect("SVD recomposition failed")
    }
    
}

/**
 * Compact Fundamental Matrix Computation, Kanatani and Sugaya 
 */
pub fn optimal_correction(initial_F: &Fundamental) -> Fundamental {
    panic!("TODO")
}