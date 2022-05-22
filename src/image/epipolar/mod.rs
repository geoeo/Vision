extern crate nalgebra as na;

mod five_point;

use na::{Vector3, Matrix3,Matrix,Dynamic, VecStorage, dimension::U9};
use crate::sensors::camera::Camera;
use crate::Float;
use crate::image::features::{Feature,Match, ImageFeature, condition_matches};
use crate::numerics::pose::optimal_correction_of_rotation;

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;

#[derive(Clone,Copy)]
pub enum EssentialDecomposition {
    FÖRSNTER,
    KANATANI
}

#[derive(Clone,Copy)]
pub enum BifocalType {
    ESSENTIAL,
    FUNDAMENTAL
} 

pub fn extract_matches<T: Feature>(matches: &Vec<Match<T>>, pyramid_scale: Float, normalize: bool) -> Vec<Match<ImageFeature>> {
    match normalize {
        true => {
            condition_matches(matches)
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
#[allow(non_snake_case)]
pub fn five_point_essential<T: Feature + Clone, C: Camera>(matches: &Vec<Match<T>>, camera_one: &C, camera_two: &C, depth_positive: bool) -> Essential {
    five_point::five_point_essential(matches,camera_one,camera_two,depth_positive)
}

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
pub fn filter_matches_from_fundamental<T: Feature + Clone>(F: &Fundamental,matches: &Vec<Match<T>>, epipiolar_thresh: Float, principal_distance_sign: Float) -> Vec<Match<T>> {
    matches.iter().filter(|m| {
            let start = m.feature_one.get_reduced_image_coordiantes(principal_distance_sign);
            let finish = m.feature_two.get_reduced_image_coordiantes(principal_distance_sign);
            let val = (start.transpose()*F*finish)[0].abs();
            val < epipiolar_thresh
        }).cloned().collect::<Vec<Match<T>>>()
}

#[allow(non_snake_case)]
pub fn filter_matches_from_motion<T: Feature + Clone, C: Camera>(matches: &Vec<Match<T>>, relative_motion: &(Vector3<Float>,Matrix3<Float>),camera_pair: &(C,C), principal_distance_sign: Float, epipiolar_thresh: Float) -> Vec<Match<T>> {
    let (cam_s,cam_f) = &camera_pair;
    let (t,R) = &relative_motion;
    let essential = essential_matrix_from_motion(t, R);
    let cam_s_inv = cam_s.get_inverse_projection();
    let cam_f_inv = cam_f.get_inverse_projection();
    let fundamental = compute_fundamental(&essential, &cam_s_inv, &cam_f_inv);

    filter_matches_from_fundamental(&fundamental,matches, epipiolar_thresh, principal_distance_sign)
}

/**
 * Computes the epipolar lines of a match.
 * Returns (line of first feature in second image, line of second feature in first image)
 */
pub fn epipolar_lines<T: Feature>(bifocal_tensor: &Matrix3<Float>, feature_match: &Match<T>, principal_distance_sign: Float) -> (Vector3<Float>, Vector3<Float>) {
    let f_from = feature_match.feature_one.get_reduced_image_coordiantes(principal_distance_sign);
    let f_to = feature_match.feature_two.get_reduced_image_coordiantes(principal_distance_sign);

    ((f_from.transpose()*bifocal_tensor).transpose(), bifocal_tensor*f_to)
}

/**
 * Photogrammetric Computer Vision p.583
 * @TODO: unify principal distance into enum
 */
#[allow(non_snake_case)]
pub fn decompose_essential_förstner<T : Feature>(
    E: &Essential, matches: &Vec<Match<T>>,
    inverse_camera_matrix_start: &Matrix3::<Float>,
    inverse_camera_matrix_finish: &Matrix3::<Float>) -> (Vector3<Float>, Matrix3<Float>,Matrix3<Float> ) {
    assert!(matches.len() > 0);
    let svd = E.svd(true,true);

    let u = &svd.u.expect("SVD failed on E");
    let v_t = &svd.v_t.expect("SVD failed on E");

    let W = Matrix3::<Float>::new(0.0, 1.0, 0.0,
                                 -1.0, 0.0 ,0.0,
                                  0.0, 0.0, 1.0);

    // let Z = Matrix3::<Float>::new(0.0, 1.0, 0.0,
    //                              -1.0, 0.0 ,0.0,
    //                               0.0, 0.0, 0.0);

    let U_norm = u*u.determinant();
    let V = v_t.transpose();
    let V_norm = V*V.determinant();

    let e_corrected = U_norm* Matrix3::<Float>::new(1.0, 0.0, 0.0,
                                            0.0, 1.0 ,0.0,
                                            0.0, 0.0, 0.0)*V_norm.transpose();



    // let Sb = u * Z * u.transpose();
    // let b = Vector3::<Float>::new(Sb[(2, 1)],Sb[(0, 2)], Sb[(1,0)]);
    let b = u.column(2).into_owned(); // / u.column(2).norm();
    let principal_distance_sign = match inverse_camera_matrix_start[(0,0)] {
        v if v < 0.0 => -1.0,
        _ => 1.0
    };
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
            let f_start = inverse_camera_matrix_start*m.feature_one.get_reduced_image_coordiantes(principal_distance_sign);
            let f_finish = inverse_camera_matrix_finish*m.feature_two.get_reduced_image_coordiantes(principal_distance_sign);

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
    
    // Might be this method five r_t after all
    rotation = optimal_correction_of_rotation(&rotation.transpose());
    
    (translation,rotation,e_corrected)

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

//TODO: refactor to only take initial cam and then list of other cams
pub fn compute_initial_cam_motions<C : Camera + Copy,T : Feature + Clone>(
        all_matches: &Vec<Vec<Match<T>>>,
        camera_data: &Vec<((usize, C),(usize,C))>,
        pyramid_scale:Float, 
        epipiolar_thresh: Float, 
        positive_principal_distance: bool,
        normalize_features: bool,
        epipolar_alg: BifocalType,
        decomp_alg: EssentialDecomposition) 
    ->  Vec<(u64,(Vector3<Float>,Matrix3<Float>))> {
    let feature_machtes =  all_matches.iter().map(|m| extract_matches(m, pyramid_scale, normalize_features)).collect::<Vec<Vec<Match<ImageFeature>>>>();
    let c_init = camera_data[0].0.1;
    let initial_motion_decomp = feature_machtes.iter().enumerate().map(|(i,m)| (m,camera_data[i])).scan((0,c_init,(Vector3::<Float>::zeros(),Matrix3::<Float>::identity())),|state, (m, (_,(id2,c2)))| {
        let (_,c_curr,(t_curr,R_curr)) = *state;
        let principal_distance_sign = match positive_principal_distance {
            true => 1.0,
            _ => -1.0
        };
        let (e,f_m) = match epipolar_alg {
            BifocalType::FUNDAMENTAL => {
                let f = eight_point(m, positive_principal_distance);
                let filtered =  filter_matches_from_fundamental(&f,m,epipiolar_thresh, principal_distance_sign);
                (compute_essential(&f,&c_curr.get_projection(),&c2.get_projection()), filtered)
            },
            BifocalType::ESSENTIAL => {
                let e = five_point_essential(m, &c_curr, &c2, positive_principal_distance);
                let f = compute_fundamental(&e, &c_curr.get_inverse_projection(), &c2.get_inverse_projection());
                let filtered =  filter_matches_from_fundamental(&f,m,epipiolar_thresh, principal_distance_sign);
                (e, filtered)
            }
        };

        let (h,rotation,_) = match decomp_alg {
            EssentialDecomposition::FÖRSNTER => decompose_essential_förstner(&e,&f_m,&c_curr.get_inverse_projection(),&c2.get_inverse_projection()),
            EssentialDecomposition::KANATANI => decompose_essential_kanatani(&e,&f_m, positive_principal_distance)
        };

        Some((id2 as u64, c2,(rotation*t_curr + h,rotation*R_curr)))
    }).map(|(id,_,motion)|(id,motion)).collect::<Vec<(u64,(Vector3<Float>,Matrix3<Float>))>>();

    initial_motion_decomp
}
