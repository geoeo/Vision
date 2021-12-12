extern crate nalgebra as na;


use na::{Vector2,Vector3, Matrix3,Matrix,Dynamic, U9, VecStorage};
use crate::Float;
use crate::image::features::{Feature,Match};

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

pub fn exatct_matches<T: Feature>(matches: &Vec<Match<T>>, pyramid_scale: Float, normalize: bool) -> Vec<(Vector2<Float>,Vector2<Float>)> {

    match normalize {
        true => {
            let number_of_matches = matches.len() as Float;
            let (left_x_acc,left_y_acc,right_x_acc,right_y_acc) = matches.iter().fold((0.0,0.0,0.0,0.0), | (u_x,u_y,v_x,v_y), f| {
                let (x_left, y_left) = f.feature_one.reconstruct_original_coordiantes_for_float(pyramid_scale);
                let (x_right, y_right) = f.feature_two.reconstruct_original_coordiantes_for_float(pyramid_scale);
                (u_x + x_left,u_y + y_left, v_x + x_right, v_y + y_right)
            });
        
            let left_x_center  = left_x_acc/ number_of_matches;
            let left_y_center  = left_y_acc/ number_of_matches;
            let right_x_center  = right_x_acc/ number_of_matches;
            let right_y_center  = right_y_acc/ number_of_matches;
        
            //Transform points so that centroid is at the origin
            let centered_features = matches.iter().map(|f| {
                let (x_left, y_left) = f.feature_one.reconstruct_original_coordiantes_for_float(pyramid_scale);
                let (x_right, y_right) = f.feature_two.reconstruct_original_coordiantes_for_float(pyramid_scale);
                ((x_left - left_x_center, y_left - left_y_center),(x_right - right_x_center, y_right - right_y_center))
            }
            ).collect::<Vec<((Float,Float),(Float,Float))>>();
        
            let (avg_distance_left,avg_distance_right) = centered_features.iter().fold((0.0,0.0), | (one_avg,two_avg), ((one_x_c,one_y_c),(two_x_c,two_y_c))| 
                (one_avg + (one_x_c.powi(2)+one_y_c.powi(2)).sqrt(), two_avg + (two_x_c.powi(2)+two_y_c.powi(2)).sqrt())
            );
        
            let scale_left = number_of_matches*(2.0 as Float).sqrt()/avg_distance_left;
            let scale_right = number_of_matches*(2.0 as Float).sqrt()/avg_distance_right;
        
            //Scale so that the average distance from the origin is sqrt(2)
            centered_features.iter().map(|((left_x_c,left_y_c),(right_x_c,right_y_c))| 
            (Vector2::<Float>::new(left_x_c*scale_left,left_y_c*scale_left),Vector2::<Float>::new(right_x_c*scale_right,right_y_c*scale_right))
            ).collect::<Vec<(Vector2<Float>,Vector2<Float>)>>()
    
        },
        false => {
                matches.iter().map(|feature| {
                    let (r_x, r_y) = feature.feature_two.reconstruct_original_coordiantes_for_float(pyramid_scale);
                    let (l_x, l_y) = feature.feature_one.reconstruct_original_coordiantes_for_float(pyramid_scale);
                    (Vector2::<Float>::new(l_x,l_y),Vector2::<Float>::new(r_x,r_y))
                }).collect()

        }
    }

}

/**
 * Photogrammetric Computer Vision p.570
 */
pub fn eight_point(matches: &Vec<(Vector2<Float>,Vector2<Float>)>) -> Fundamental {
    let number_of_matches = matches.len() as Float;
    assert!(number_of_matches >= 8.0);


    let mut A = Matrix::<Float,Dynamic, U9, VecStorage<Float,Dynamic,U9>>::zeros(matches.len());
    for i in 0..A.nrows() {

        let feature_right = matches[i].1;
        let feature_left = matches[i].0;

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
    let mut min_idx = svd.singular_values.imin();
    let v_t =  &svd.v_t.expect("SVD failed on A");
    let f = &v_t.row(min_idx);
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
    min_idx = svd_f.singular_values.imin();
    let mut acc = 0.0;
    for i in 0..2 {
        if i == min_idx{
            svd_f.singular_values[i] = 0.0;
        } else {
            acc += svd_f.singular_values[i].powi(2);
        }

    }
    svd_f.singular_values[min_idx] = 0.0;
    svd_f.singular_values /= acc.sqrt();
    svd_f.recompose().ok().expect("SVD recomposition failed")
}

pub fn compute_essential(F: &Fundamental, projection_left: &Matrix3<Float>, projection_right: &Matrix3<Float>) -> Essential {
    projection_left.transpose()*F*projection_right
}

pub fn filter_matches(F: &Fundamental,matches: &Vec<(Vector2<Float>,Vector2<Float>)>) -> Vec<(Vector3<Float>,Vector3<Float>)> {
    matches.iter().map(|(l,r)| {
    
            let feature_left = Vector3::new(l[0],l[1],1.0);
            let feature_right = Vector3::new(r[0],r[1],1.0);
            (feature_left,feature_right)
        }).filter(|(l,r)| 
            (l.transpose()*F*r)[0].abs() < 1.0
        ).collect::<Vec<(Vector3<Float>,Vector3<Float>)>>()
}

/**
 * Photogrammetric Computer Vision p.583
 */
pub fn decompose_essential_förstner(E: &Essential,matches: &Vec<(Vector3<Float>,Vector3<Float>)>) -> (Vector3<Float>, Matrix3<Float>) {
    let svd = E.svd(true,true);
    let min_idx = svd.singular_values.imin();
    let u = &svd.u.expect("SVD failed on E");
    let v_t = &svd.v_t.expect("SVD failed on E");
    let mut h = Vector3::<Float>::from_columns(&[u.column(min_idx)]);

    let W = Matrix3::<Float>::new(0.0, 1.0, 0.0,
                                 -1.0, 0.0 ,0.0,
                                  0.0, 0.0, 1.0);

    let U_norm = u*u.determinant();
    let V_norm = v_t.transpose()*v_t.transpose().determinant();
    let w_matrices = vec!(W,W.transpose(), -W, (-W).transpose());
    let h_vecs = vec!(h,h, -h, -h);
    let R_matrices = vec!(V_norm*w_matrices[0]*U_norm.transpose(),V_norm*w_matrices[1]*U_norm.transpose(), V_norm*w_matrices[0]*U_norm.transpose(), V_norm*w_matrices[1]*U_norm.transpose());

    let mut translation = Vector3::<Float>::zeros();
    let mut rotation = Matrix3::<Float>::identity();
    for i in 0..4 {
        let W = w_matrices[i];
        let h = h_vecs[i];
        let R = R_matrices[i];
        let mut v_sign = 0.0;
        let mut u_sign = 0.0;
        for (feature_left,feature_right) in matches {

            let fl = Vector3::<Float>::new(feature_left[0],feature_left[1],-1.0);
            let fr = Vector3::<Float>::new(feature_right[0],feature_right[1],-1.0);

            let binormal = ((h.cross_matrix()*fl).cross_matrix()*h).normalize();
            let mat = Matrix3::<Float>::from_columns(&[h,binormal,fl.cross_matrix()*R.transpose()*fr]);
            let s_i = mat.determinant();
            let s_i_sign = match s_i {
                det if det > 0.0 => 1.0,
                det if det < 0.0 => -1.0,
                _ => 0.0
            };
            v_sign += s_i_sign;
            let s_r = (binormal.transpose()*R.transpose()*fr)[0];
            u_sign += match s_i_sign*s_r {
                s if s > 0.0 => 1.0,
                s if s < 0.0 => -1.0,
                _ => 0.0
            }
        }

        let u_sign_avg = u_sign /matches.len() as Float;
        let v_sign_avg = v_sign /matches.len() as Float;

        if u_sign_avg > 0.0 && v_sign_avg > 0.0 {
            translation = h;
            rotation = R;
            break;
        }
    }

    (translation,rotation)

}

/**
 * Statistical Optimization for Geometric Computation p.338
 */
pub fn decompose_essential_kanatani(E: &Essential, matches: &Vec<(Vector3<Float>,Vector3<Float>)>) -> (Vector3<Float>, Matrix3<Float>) {
    let mut translation = Vector3::<Float>::zeros();
    let mut R = Matrix3::<Float>::identity();

    if matches.len() > 0 {
        let svd = (E*E.transpose()).svd(true,false);
        let min_idx = svd.singular_values.imin();
        let u = &svd.u.expect("SVD failed on E");
        let mut h = u.column(min_idx).normalize();
        let sum_of_determinants = matches.iter().fold(0.0, |acc,(l,r)| {
            let mat = Matrix3::from_columns(&[h,*l,E*r]);
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
        R = svd_k.recompose().ok().expect("SVD recomposition failed on K");
        translation = h;

    }

    (translation,R.transpose())

}