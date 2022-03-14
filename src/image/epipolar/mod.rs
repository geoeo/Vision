extern crate nalgebra as na;


use na::{Vector3, Matrix3,Matrix,Dynamic, U9, VecStorage};
use crate::sensors::camera::Camera;
use crate::Float;
use crate::image::features::{Feature,Match, ImageFeature};

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

#[derive(Clone,Copy)]
pub enum EssentialDecomposition {
    FÖRSNTER,
    KANATANI
} 

pub fn extract_matches<T: Feature>(matches: &Vec<Match<T>>, pyramid_scale: Float, normalize: bool) -> Vec<Match<ImageFeature>> {

    // Seems to amplify errors when mismatches are present
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
            Match { feature_one: ImageFeature::new(left_x_c*scale_left,left_y_c*scale_left), feature_two: ImageFeature::new(right_x_c*scale_right,right_y_c*scale_right)}
            ).collect::<Vec<Match<ImageFeature>>>()
    
        },
        false => {
                matches.iter().map(|feature| {
                    let (r_x, r_y) = feature.feature_two.reconstruct_original_coordiantes_for_float(pyramid_scale);
                    let (l_x, l_y) = feature.feature_one.reconstruct_original_coordiantes_for_float(pyramid_scale);
                    Match { feature_one: ImageFeature::new(l_x,l_y), feature_two: ImageFeature::new(r_x,r_y)}
                }).collect()

        }
    }

}

/**
 * Photogrammetric Computer Vision p.570
 */
#[allow(non_snake_case)]
pub fn eight_point<T : Feature>(matches: &Vec<Match<T>>) -> Fundamental {
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
    svd_f.recompose().ok().expect("SVD recomposition failed")
}

//TODO: write a test for this
pub fn essential_matrix_from_motion(translation: &Vector3<Float>, rotation: &Matrix3<Float>) -> Matrix3<Float> {
    translation.cross_matrix()*rotation.transpose()
}

#[allow(non_snake_case)]
pub fn compute_essential(F: &Fundamental, projection_start: &Matrix3<Float>, projection_finish: &Matrix3<Float>) -> Essential {
    projection_start.transpose()*F*projection_finish
}

#[allow(non_snake_case)]
pub fn compute_fundamental(E: &Essential, inverse_projection_start: &Matrix3<Float>, inverse_projection_finish: &Matrix3<Float>) -> Essential {
    inverse_projection_start.transpose()*E*inverse_projection_finish
}

#[allow(non_snake_case)]
pub fn filter_matches_from_fundamental<T: Feature + Clone>(F: &Fundamental,matches: &Vec<Match<T>>, epipiolar_thresh: Float) -> Vec<Match<T>> {
    matches.iter().filter(|m| {
            let start = m.feature_one.get_as_3d_point(1.0);
            let finish = m.feature_two.get_as_3d_point(1.0);
            (start.transpose()*F*finish)[0].abs() < epipiolar_thresh
        }).cloned().collect::<Vec<Match<T>>>()
}

#[allow(non_snake_case)]
pub fn filter_matches_from_motion<T: Feature + Clone, C: Camera>(matches: &Vec<Match<T>>, relative_motion: &(Vector3<Float>,Matrix3<Float>),camera_pair: &(C,C), epipiolar_thresh: Float) -> Vec<Match<T>> {
    let (cam_s,cam_f) = &camera_pair;
    let (t,R) = &relative_motion;
    let essential = essential_matrix_from_motion(t, R);
    let cam_s_inv = cam_s.get_inverse_projection();
    let cam_f_inv = cam_f.get_inverse_projection();
    let fundamental = compute_fundamental(&essential, &cam_s_inv, &cam_f_inv);

    filter_matches_from_fundamental(&fundamental,matches,epipiolar_thresh)
}


/**
 * Photogrammetric Computer Vision p.583
 */
#[allow(non_snake_case)]
pub fn decompose_essential_förstner<T : Feature>(E: &Essential,matches: &Vec<Match<T>>) -> (Vector3<Float>, Matrix3<Float>) {
    assert!(matches.len() > 0);
    let svd = E.svd(true,true);
    let min_idx = svd.singular_values.imin();
    let u = &svd.u.expect("SVD failed on E");
    let v_t = &svd.v_t.expect("SVD failed on E");
    let h = Vector3::<Float>::from_columns(&[u.column(min_idx)]);

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
        let h = h_vecs[i];
        let R = R_matrices[i];
        let mut v_sign = 0.0;
        let mut u_sign = 0.0;
        for m in matches {
            
            let f_start = m.feature_one.get_as_3d_point(1.0);
            let f_finish = m.feature_two.get_as_3d_point(1.0);

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

//TODO: this is still a little unclear depending on positive or negative depth
/**
 * Statistical Optimization for Geometric Computation p.338
 */
#[allow(non_snake_case)]
pub fn decompose_essential_kanatani<T: Feature>(E: &Essential, matches: &Vec<Match<T>>, is_depth_positive: bool) -> (Vector3<Float>, Matrix3<Float>) {
    assert!(matches.len() > 0);

    let svd = (E*E.transpose()).svd(true,false);
    let min_idx = svd.singular_values.imin();
    let u = &svd.u.expect("SVD failed on E");
    let mut h = u.column(min_idx).normalize();
    let sum_of_determinants = matches.iter().fold(0.0, |acc,m| {

        let (start_new,finish_new) = (m.feature_one.get_as_3d_point(1.0),m.feature_one.get_as_3d_point(1.0));

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

    
    match is_depth_positive {
        false => (-R.transpose()*translation,R.transpose()),
        true => (translation,R)
    }
    

}

pub fn compute_initial_cam_motions<C : Camera + Copy,T : Feature + Clone>(all_matches: &Vec<Vec<Match<T>>>,camera_data: &Vec<((usize, C),(usize,C))>,pyramid_scale:Float, epipiolar_thresh: Float, is_depth_positive: bool,decomp_alg: EssentialDecomposition) 
    ->  (Vec<(u64,(Vector3<Float>,Matrix3<Float>))>,Vec<Vec<Match<ImageFeature>>>) {
    let feature_machtes = all_matches.iter().filter(|m| m.len() >= 8).map(|m| extract_matches(m, pyramid_scale, false)).collect::<Vec<Vec<Match<ImageFeature>>>>();
    let fundamental_matrices = feature_machtes.iter().map(|m| eight_point(m)).collect::<Vec<Fundamental>>();
    let accepted_matches = fundamental_matrices.iter().zip(feature_machtes.iter()).map(|(f,m)| filter_matches_from_fundamental(f, m,epipiolar_thresh)).collect::<Vec<Vec<Match<ImageFeature>>>>();
    let essential_matrices = fundamental_matrices.iter().enumerate().map(|(i,f)| {
        let ((id1,c1),(id2,c2)) = camera_data[i];
        (id1,id2,compute_essential(f, &c1.get_projection(), &c2.get_projection()))
        
    }).collect::<Vec<(usize,usize,Essential)>>();

    let initial_motion_decomp = essential_matrices.iter().filter(|(id1,_,_)| *id1 == camera_data[0].0.0).enumerate().map(|(i,(_,id2,e))| {
        let matches = &accepted_matches[i];

        let (h,rotation) = match (decomp_alg,matches.len()) {
            (_,count) if count < 8 => (Vector3::<Float>::zeros(), Matrix3::<Float>::identity()),
            (EssentialDecomposition::FÖRSNTER,_) => decompose_essential_förstner(e,matches),
            (EssentialDecomposition::KANATANI,_) => decompose_essential_kanatani(e,matches, is_depth_positive)
        };

        (*id2 as u64,(h,rotation))
    }).collect::<Vec<(u64,(Vector3<Float>,Matrix3<Float>))>>();

    (initial_motion_decomp,accepted_matches)
}
