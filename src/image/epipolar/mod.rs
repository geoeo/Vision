extern crate nalgebra as na;


use na::{Matrix3,Matrix,Dynamic, U9, VecStorage};
use crate::Float;
use crate::image::features::{Feature,Match};


pub fn eight_point<T: Feature>(matches: &Vec<Match<T>>) -> Matrix3<Float> {
    let number_of_matches = matches.len() as Float;


    let (one_x_acc,one_y_acc,two_x_acc,two_y_acc) = matches.iter().fold((0,0,0,0), | (u_x,u_y,v_x,v_y), f| {
        (u_x + f.feature_one.1.get_x_image(),u_y + f.feature_one.1.get_y_image(), v_x + f.feature_two.1.get_x_image(), v_y + f.feature_two.1.get_y_image())
    });

    let one_x_center  = (one_x_acc as Float)/ number_of_matches;
    let one_y_center  = (one_y_acc as Float)/ number_of_matches;
    let two_x_center  = (two_x_acc as Float)/ number_of_matches;
    let two_y_center  = (two_y_acc as Float)/ number_of_matches;

    //Transform points so that centroid is at the origin
    let centered_features = matches.iter().map(|f| 
        ((f.feature_one.1.get_x_image() as Float - one_x_center, f.feature_one.1.get_y_image() as Float - one_y_center),(f.feature_two.1.get_x_image() as Float - two_x_center, f.feature_two.1.get_y_image() as Float - two_y_center))
    ).collect::<Vec<((Float,Float),(Float,Float))>>();



    let (avg_distance_one,avg_distance_two) = centered_features.iter().fold((0.0,0.0), | (one_avg,two_avg), ((one_x_c,one_y_c),(two_x_c,two_y_c))| 
        (one_avg + (one_x_c.powi(2)+one_y_c.powi(2)).sqrt(), two_avg + (two_x_c.powi(2)+two_y_c.powi(2)).sqrt())
    );

    let scale_one = number_of_matches*(2.0 as Float).sqrt()/avg_distance_one;
    let scale_two = number_of_matches*(2.0 as Float).sqrt()/avg_distance_two;

    //Scale so that the average distance from the origin is sqrt(2)
    let normalized_features = centered_features.iter().map(|((one_x_c,one_y_c),(two_x_c,two_y_c))| 
        ((one_x_c*scale_one,one_y_c*scale_one),(two_x_c*scale_two,two_y_c*scale_two))
    ).collect::<Vec<((Float,Float),(Float,Float))>>();


    let mut A = Matrix::<Float,Dynamic, U9, VecStorage<Float,Dynamic,U9>>::zeros(normalized_features.len());
    for i in 0..A.nrows() {
        let ((u,v),(u_prime,v_prime)) = normalized_features[i];

        A[(i,0)] = u*u_prime;
        A[(i,1)] = u*v_prime;
        A[(i,2)] = u;
        A[(i,3)] = v*u_prime;
        A[(i,4)] = v*v_prime;
        A[(i,5)] = v;
        A[(i,6)] = u_prime;
        A[(i,7)] = v_prime;
        A[(i,8)] = 1.0;
    }


    let svd = A.svd(false,true);
    let s =  &svd.singular_values;
    let mut min_idx = 0;
    let mut min_val = s[0];
    for i in 1..s.nrows() {
        if s[i] < min_val {
            min_val = s[i];
            min_idx = i;
        } 
    }

    let v_t =  svd.v_t.expect("SVD failed on A");
    let f = v_t.row(min_idx);
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
    let s_f =  &svd_f.singular_values;
    min_idx = 0;
    min_val = s_f[0];
    for i in 1..s_f.nrows() {
        if s_f[i] < min_val {
            min_val = s[i];
            min_idx = i;
        } 
    }

    svd_f.singular_values[min_idx] = 0.0;

    svd_f.recompose().ok().expect("SVD recomposition failed")
}