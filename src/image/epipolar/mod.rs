extern crate nalgebra as na;
extern crate rand;

pub mod tensor;

use na::{Vector3, Matrix3};

use crate::sensors::camera::Camera;
use crate::Float;
use crate::image::features::{Feature,Match, ImageFeature, condition_matches};
use crate::sfm::SFMConfig;

pub type Fundamental =  Matrix3<Float>;
pub type Essential =  Matrix3<Float>;


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
pub fn filter_matches_from_motion<T: Feature + Clone, C: Camera<Float>>(matches: &Vec<Match<T>>, relative_motion: &(Vector3<Float>,Matrix3<Float>),camera_pair: &(C,C), epipiolar_thresh: Float) -> Vec<Match<T>> {
    let (cam_s,cam_f) = &camera_pair;
    let (t,R) = &relative_motion;
    let essential = tensor::essential_matrix_from_motion(t, R);
    let cam_s_inv = cam_s.get_inverse_projection();
    let cam_f_inv = cam_f.get_inverse_projection();
    let fundamental = tensor::compute_fundamental(&essential, &cam_s_inv, &cam_f_inv);

    tensor::filter_matches_from_fundamental(&fundamental,matches, epipiolar_thresh, cam_s,cam_f)
}

/**
 * Computes the epipolar lines of a match.
 * Returns (line of first feature in second image, line of second feature in first image)
 */
pub fn epipolar_lines<T: Feature>(bifocal_tensor: &Matrix3<Float>, feature_match: &Match<T>, cam_one_intrinsics: &Matrix3<Float>, cam_two_intrinsics: &Matrix3<Float>) -> (Vector3<Float>, Vector3<Float>) {
    //TODO: pass in camera object not intrinsics matrix
    let f_from = feature_match.feature_one.get_camera_ray(cam_one_intrinsics);
    let f_to = feature_match.feature_two.get_camera_ray(cam_two_intrinsics);

    ((f_from.transpose()*bifocal_tensor).transpose(), bifocal_tensor*f_to)
}

#[allow(non_snake_case)]
pub fn compute_pairwise_cam_motions_with_filtered_matches_for_path<C : Camera<Float> + Copy, C2, T : Feature + Clone>(
        sfm_config: &SFMConfig<C,C2, T>,
        path_idx: usize,
        pyramid_scale:Float, 
        epipolar_thresh: Float, 
        normalize_features: bool,
        epipolar_alg: tensor::BifocalType,
        decomp_alg: tensor::EssentialDecomposition) 
    ->  (Vec<(usize,(Vector3<Float>,Matrix3<Float>))>,Vec<Vec<Match<ImageFeature>>>) {
    let root_id = sfm_config.root();
    let path = &sfm_config.paths()[path_idx];
    let matches = &sfm_config.matches()[path_idx];
    let filtered_matches_by_track = &sfm_config.filtered_matches_by_tracks()[path_idx];
    let camera_map = sfm_config.camera_map();
    let root_cam = camera_map.get(&root_id).expect("compute_pairwise_cam_motions_for_path: could not get root cam");
    let feature_machtes =  matches.iter().map(|m| extract_matches(m, pyramid_scale, normalize_features)).collect::<Vec<Vec<Match<ImageFeature>>>>();
    let filtered_feature_machtes_by_track =  filtered_matches_by_track.iter().map(|m| extract_matches(m, pyramid_scale, normalize_features)).collect::<Vec<Vec<Match<ImageFeature>>>>();
    feature_machtes.iter().zip(filtered_feature_machtes_by_track.iter()).enumerate().map(|(i,(m,f_m_tracks))| {
        let c1 = match i {
            0 => root_cam,
            idx => camera_map.get(&path[idx-1]).expect("compute_pairwise_cam_motions_for_path: could not get previous cam")
        };
        let id2 = path[i];
        let c2 = camera_map.get(&id2).expect("compute_pairwise_cam_motions_for_path: could not get second camera");
        let (e,f_m) = match epipolar_alg {
            tensor::BifocalType::FUNDAMENTAL => {
                let f = tensor::fundamental::eight_point(f_m_tracks, false); //TODO: make this configurable
                let filtered =  tensor::filter_matches_from_fundamental(&f,m,epipolar_thresh, c1,c2);
                (tensor::compute_essential(&f,&c1.get_projection(),&c2.get_projection()), filtered)
            },
            tensor::BifocalType::ESSENTIAL => {
                //TODO: put these in configs
                //Do NcR for
                let e = tensor::ransac_five_point_essential(f_m_tracks, c1, c2, 2.0,1e5 as usize, 5 );
                //let e = five_point_essential(f_m_tracks, c1, c2);
                let f = tensor::compute_fundamental(&e, &c1.get_inverse_projection(), &c2.get_inverse_projection());
                let filtered =  tensor::filter_matches_from_fundamental(&f,m,epipolar_thresh,c1,c2);
                (e, filtered)
            }
        };

        let (h,rotation,_) = match decomp_alg {
            tensor::EssentialDecomposition::FÖRSNTER => tensor::decompose_essential_förstner(&e,&f_m,c1,c2),
            tensor::EssentialDecomposition::KANATANI => tensor::decompose_essential_kanatani(&e,&f_m, false)
        };
        let new_state = (id2,(h, rotation));
        (new_state, f_m)
    }).unzip()
}


#[allow(non_snake_case)]
pub fn compute_pairwise_cam_motions_with_filtered_matches<C: Camera<Float> + Copy, C2, T : Feature + Clone>(
        sfm_config: &SFMConfig<C, C2, T>,
        pyramid_scale:Float, 
        epipolar_thresh: Float, 
        normalize_features: bool,
        epipolar_alg: tensor::BifocalType,
        decomp_alg: tensor::EssentialDecomposition) 
    ->  (Vec<Vec<(usize,(Vector3<Float>,Matrix3<Float>))>>,Vec<Vec<Vec<Match<ImageFeature>>>>) {
    (0..sfm_config.paths().len()).map(|i| 
        compute_pairwise_cam_motions_with_filtered_matches_for_path(
        sfm_config,
        i,
        pyramid_scale,
        epipolar_thresh,
        normalize_features,
        epipolar_alg, 
        decomp_alg)
    ).unzip()
    
}
