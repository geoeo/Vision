extern crate nalgebra as na;
extern crate rand;

mod five_point;
pub mod fundamental;

use na::{Vector3, Matrix3, SMatrix};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::iter::FromIterator;

use crate::sensors::camera::Camera;
use crate::Float;
use crate::image::features::{Feature,Match};
use crate::numerics::pose::optimal_correction_of_rotation;

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

#[derive(Clone,Copy)]
pub enum BifocalType {
    ESSENTIAL,
    FUNDAMENTAL
}

#[derive(Clone,Copy)]
pub enum EssentialDecomposition {
    FÖRSNTER,
    KANATANI
}

#[allow(non_snake_case)]
pub fn filter_matches_from_fundamental<T: Feature + Clone>(F: &Fundamental,matches: &Vec<Match<T>>, epipiolar_thresh: Float) -> Vec<Match<T>> {
    matches.iter().filter(|m| {
            let start = m.feature_one.get_as_3d_point(-1.0);
            let finish = m.feature_two.get_as_3d_point(-1.0);
            let val = (start.transpose()*F*finish)[0].abs();
            val < epipiolar_thresh
        }).cloned().collect::<Vec<Match<T>>>()
}

//TODO: Investigate why sorting the matched changes the solver convergence -> using set for now
#[allow(non_snake_case)]
pub fn select_best_matches_from_fundamental<T: Feature + Clone>(F: &Fundamental,matches: &Vec<Match<T>>, perc: Float) -> Vec<Match<T>> {
    assert! (0.0 <= perc && perc <= 1.0);
    let num = matches.len();
    let take_num = (perc*(num as Float)) as usize;
    let mut sorted_indices = matches.iter().enumerate().map(|(i,m)| {
            let start = m.feature_one.get_as_3d_point(-1.0);
            let finish = m.feature_two.get_as_3d_point(-1.0);
            (i , (start.transpose()*F*finish)[0].abs())
        }).collect::<Vec<(usize,Float)>>();

        sorted_indices.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let accepted_indices = HashSet::<usize>::from_iter(sorted_indices.iter().map(|(i,_)| i.clone()));
        matches.iter().enumerate().filter(|(i,_)| accepted_indices.contains(i)).take(take_num).map(|(_,m)| m.clone()).collect::<Vec<Match<T>>>()
}


pub fn ransac_five_point_essential<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, camera_one: &C, camera_two: &C, epipolar_thresh: Float, ransac_it: usize, ransac_size: usize) -> Essential {
    let mut max_inlier_count = 0;
    let mut min_det = Float::INFINITY;
    let mut min_singular_value = Float::INFINITY;
    let mut best_essential: Option<Essential> = None;
    for _ in 0..ransac_it {
        let samples: Vec<_> = matches.choose_multiple(&mut rand::thread_rng(), ransac_size).map(|x| x.clone()).collect();
        let essential_option = five_point::five_point_essential(&samples,camera_one,camera_two);
        match essential_option {
            Some(essential) => {
                let svd = essential.svd(false,false);
                let min_val = svd.singular_values[2];
                let f = compute_fundamental(&essential, &camera_one.get_inverse_projection(), &camera_two.get_inverse_projection());
                best_essential = match (min_val.abs(), essential.determinant().abs(), filter_matches_from_fundamental(&f,matches,epipolar_thresh).len()) {
                    (singular_val, det, inliers) if (inliers > max_inlier_count) || (inliers == max_inlier_count && singular_val < min_singular_value) => {
                        max_inlier_count = inliers;
                        min_singular_value = singular_val;
                        min_det = det;
                        Some(essential)
                    },
                    _ => best_essential
                };
            },
            None => ()
        };
    }

    println!("Best inliner count for essential matrix was {} out of {} matches. That is {} % with det: {}", max_inlier_count, matches.len(), ((max_inlier_count as Float) / (matches.len() as Float)) * 100.0, min_det);
    best_essential.expect("No essential matrix could be computer via RANSAC")
}


pub fn five_point_essential<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, camera_one: &C, camera_two: &C) -> Essential {
    five_point::five_point_essential(&matches,camera_one,camera_two).expect("five_point_essential: failed")
}

pub fn essential_matrix_from_motion(translation: &Vector3<Float>, rotation: &Matrix3<Float>) -> Essential {
    translation.cross_matrix()*rotation.transpose()
}

#[allow(non_snake_case)]
pub fn compute_essential(F: &Fundamental, projection_start: &Matrix3<Float>, projection_finish: &Matrix3<Float>) -> Essential {
    projection_start.transpose()*F*projection_finish
}

#[allow(non_snake_case)]
pub fn compute_covariance_of_essential_for_eight_matches(fundamental_cov: &SMatrix<Float, 9, 9>, projection_start: &Matrix3<Float>, projection_finish: &Matrix3<Float>) -> SMatrix<Float, 9, 9>  {
    let factor = projection_finish.transpose().kronecker(projection_start);
    factor*fundamental_cov*factor.transpose()
}

#[allow(non_snake_case)]
pub fn compute_fundamental(E: &Essential, inverse_projection_start: &Matrix3<Float>, inverse_projection_finish: &Matrix3<Float>) -> Essential {
    inverse_projection_start.transpose()*E*inverse_projection_finish
}

/**
 * Photogrammetric Computer Vision p.583
 * @TODO: unify principal distance into enum
 */
#[allow(non_snake_case)]
pub fn decompose_essential_förstner<T : Feature, C : Camera<Float>>(
    E: &Essential, matches: &Vec<Match<T>>,
    camera_start: &C,
    camera_finish: &C) -> (Vector3<Float>, Matrix3<Float>,Matrix3<Float> ) {
    assert!(matches.len() > 0);

    let inverse_camera_matrix_start = camera_start.get_inverse_projection();
    let inverse_camera_matrix_finish = camera_finish.get_inverse_projection();

    let svd = E.svd(true,true);
    let u = &svd.u.expect("SVD failed on E");
    let v_t = &svd.v_t.expect("SVD failed on E");

    let W = Matrix3::<Float>::new(0.0, 1.0, 0.0,
                                 -1.0, 0.0 ,0.0,
                                  0.0, 0.0, 1.0);

    let U_norm = u*u.determinant();
    let V = v_t.transpose();
    let V_norm = V*V.determinant();

    let e_corrected = U_norm* Matrix3::<Float>::new(1.0, 0.0, 0.0,
                                            0.0, 1.0 ,0.0,
                                            0.0, 0.0, 0.0)*V_norm.transpose();


    let b = u.column(2).into_owned(); 
    let R_matrices = vec!(V_norm*W*U_norm.transpose(), V_norm*W*U_norm.transpose(),V_norm*W.transpose()*U_norm.transpose(), V_norm*W.transpose()*U_norm.transpose());
    let h_vecs = vec!(b,-b, b, -b);

    let mut translation = Vector3::<Float>::zeros();
    let mut rotation = Matrix3::<Float>::identity();
    for i in 0..4 {
        let h = h_vecs[i];
        let R = R_matrices[i];
        let mut v_sign = 0.0;
        let mut u_sign = 0.0;
        for m in matches {
            let f_start = m.feature_one.get_camera_ray(&inverse_camera_matrix_start);
            let f_finish = m.feature_two.get_camera_ray(&inverse_camera_matrix_finish);

            let binormal = ((h.cross_matrix()*f_start).cross_matrix()*h).normalize();
            let mat = Matrix3::<Float>::from_columns(&[h,binormal,f_start.cross_matrix()*R.transpose()*f_finish]);
            let s_i = mat.determinant();
            let s_i_sign = match s_i {

                det if det > 0.0 => 1.0,
                det if det < 0.0 => -1.0,
                _ => 0.0
            };
            v_sign += s_i_sign;
            let s_r = (binormal.transpose()*R.transpose()*f_finish)[0];
            let s_r_sign = match s_r {
                s if s > 0.0 => 1.0,
                s if s < 0.0 => -1.0,
                _ => 0.0
            };
            u_sign += match s_i_sign*s_r_sign {
                s if s > 0.0 => 1.0,
                s if s < 0.0 => -1.0,
                _ => 0.0
            };
        }

        let u_sign_avg = u_sign /matches.len() as Float; 
        let v_sign_avg = v_sign /matches.len() as Float;

        if u_sign_avg > 0.0 && v_sign_avg > 0.0 {
            translation = h;
            rotation = R;
            break;
        } 
    }
    
    (translation,optimal_correction_of_rotation(&rotation),e_corrected)

}

//TODO: this is still a little unclear depending on positive or negative depth
/**
 * Statistical Optimization for Geometric Computation p.338
 */
#[allow(non_snake_case)]
pub fn decompose_essential_kanatani<T: Feature>(E: &Essential, matches: &Vec<Match<T>>, is_depth_positive: bool) -> (Vector3<Float>, Matrix3<Float>, Matrix3<Float>) {
    assert!(matches.len() > 0);
    assert!(!is_depth_positive);
    println!("WARN: decompose_essential_kanatani is buggy");
    let svd = (E*E.transpose()).svd(true,false);
    let min_idx = svd.singular_values.imin();
    let u = &svd.u.expect("SVD failed on E");
    let mut h = u.column(min_idx).normalize();

    let sum_of_determinants = matches.iter().fold(0.0, |acc,m| {

        let (start_new,finish_new) = (m.feature_one.get_as_3d_point(-1.0),m.feature_one.get_as_3d_point(-1.0));

        let mat = Matrix3::from_columns(&[h,start_new,E*finish_new]);
        match mat.determinant() {
            v if v > 0.0 => acc+1.0,
            v if v < 0.0 => acc-1.0,
            _ => acc
        }
    });
    if sum_of_determinants < 0.0 {
        h  *= -1.0; 
    }

    let K = (-h).cross_matrix()*E;
    let mut svd_k = K.svd(true,true);
    let u_k = svd_k.u.expect("SVD U failed on K");
    let v_t_k = svd_k.v_t.expect("SVD V_t failed on K");
    let min_idx = svd_k.singular_values.imin();
    for i in 0..svd_k.singular_values.nrows(){
        if i == min_idx {
            svd_k.singular_values[i] = (u_k*v_t_k).determinant();
        } else {
            svd_k.singular_values[i] = 1.0;
        }
    }
    let R = svd_k.recompose().ok().expect("SVD recomposition failed on K");
    let translation = h;

    //TODO: corrected E
    (translation,R, Matrix3::<Float>::identity())

}