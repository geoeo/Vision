extern crate nalgebra as na;

use na::{Vector3,Matrix3,DVector};
use std::collections::HashMap;
use crate::image::{
    features::{Feature,Match},
    features::geometry::point::Point
};
use crate::sfm::bundle_adjustment::state::State;
use crate::{Float, reconstruct_original_coordiantes_for_float};



pub struct CameraFeatureMap {
    pub camera_map: HashMap<u64, (usize, HashMap<usize,usize>)>,
    pub number_of_unique_points: usize,
    /**
     * 2d Vector of dim rows: point, dim : cam. Where the values are in (x,y) tuples
     */
    pub point_cam_map: Vec<Vec<Option<(Float,Float)>>>,
    pub image_row_col: (usize,usize)

}

impl CameraFeatureMap {

    pub fn new<T: Feature>(matches: & Vec<Vec<Match<T>>>, n_cams: usize, image_row_col: (usize,usize)) -> CameraFeatureMap {
        let max_number_of_points = matches.iter().fold(0,|acc,x| acc + x.len());
        CameraFeatureMap{
            camera_map:  HashMap::new(),
            number_of_unique_points: 0,
            point_cam_map: vec![vec![None;n_cams]; max_number_of_points],
            image_row_col
        }
    }

    pub fn add_camera(&mut self, ids: Vec<u64>, features_per_octave: usize, octave_count: usize) -> () {
        for i in 0..ids.len(){
            let id = ids[i];
            self.camera_map.insert(id,(i,HashMap::new()));
        }

    }

    fn linear_image_idx(&self, p: &Point<Float>) -> usize {
        (p.y as usize)*self.image_row_col.1+(p.x as usize)
    }


    pub fn add_feature(&mut self, source_cam_id: u64, other_cam_id: u64, 
        x_source: Float, y_source: Float, octave_index_source: usize, 
        x_other: Float, y_other: Float, octave_index_other: usize,  
        pyramid_scale: Float) -> () {

        let (x_source_recon,y_source_recon) = reconstruct_original_coordiantes_for_float(x_source, y_source, pyramid_scale, octave_index_source as i32);
        let (x_other_recon,y_other_recon) = reconstruct_original_coordiantes_for_float(x_other, y_other, pyramid_scale, octave_index_other as i32);
        let point_source = Point::<Float>::new(x_source_recon,y_source_recon);
        let point_other = Point::<Float>::new(x_other_recon,y_other_recon); 

        let point_source_idx = self.linear_image_idx(&point_source);
        let point_other_idx = self.linear_image_idx(&point_other);

        let source_cam_idx = self.camera_map.get(&source_cam_id).unwrap().0;
        let other_cam_idx = self.camera_map.get(&other_cam_id).unwrap().0;
        
        let source_point_id =  self.camera_map.get(&source_cam_id).unwrap().1.get(&point_source_idx);
        let other_point_id = self.camera_map.get(&other_cam_id).unwrap().1.get(&point_other_idx);

        match (source_point_id.clone(),other_point_id.clone()) {
            (None,None) => {
                self.point_cam_map[self.number_of_unique_points][source_cam_idx] = Some((point_source.x,point_source.y));
                self.point_cam_map[self.number_of_unique_points][other_cam_idx] = Some((point_other.x,point_other.y));
                self.camera_map.get_mut(&source_cam_id).unwrap().1.insert(point_source_idx,self.number_of_unique_points);
                self.camera_map.get_mut(&other_cam_id).unwrap().1.insert(point_other_idx, self.number_of_unique_points);

                self.number_of_unique_points += 1;
            },
            (Some(&point_id),_) => {
                self.point_cam_map[point_id][other_cam_idx] = Some((point_other.x,point_other.y));
                self.camera_map.get_mut(&other_cam_id).unwrap().1.insert(point_other_idx, point_id);

            },
            (None,Some(&point_id)) => {
                self.camera_map.get_mut(&source_cam_id).unwrap().1.insert(point_source_idx,point_id);
                self.point_cam_map[point_id][source_cam_idx] = Some((point_source.x,point_source.y));

            }
        }

    }

    pub fn add_matches<T: Feature>(&mut self, image_id_pairs: &Vec<(u64, u64)>, matches: & Vec<Vec<Match<T>>>, pyramid_scale: Float) -> () {
        assert_eq!(image_id_pairs.len(), matches.len());
        for i in 0..image_id_pairs.len(){
            let (id_a,id_b) = image_id_pairs[i];
            let matches_for_pair = &matches[i];

            for feature_match in matches_for_pair {
                let match_a = &feature_match.feature_one;
                let match_b = &feature_match.feature_two;


                self.add_feature(id_a, id_b, 
                    match_a.get_x_image_float(), match_a.get_y_image_float(),match_a.get_closest_sigma_level(),
                    match_b.get_x_image_float(), match_b.get_y_image_float(),match_b.get_closest_sigma_level(),
                    pyramid_scale);
            }

        }

    }

    /**
     * initial_motion should all be with respect to the first camera
     */
    pub fn get_initial_state(&self, initial_motions : Option<&Vec<(Vector3<Float>,Matrix3<Float>)>>, initial_depth: Float) -> State {

        let number_of_cameras = self.camera_map.keys().len();
        let number_of_unqiue_points = self.number_of_unique_points;
        let number_of_cam_parameters = 6*number_of_cameras;
        let number_of_point_parameters = 3*number_of_unqiue_points;
        let total_parameters = number_of_cam_parameters+number_of_point_parameters;
        let mut data = DVector::<Float>::zeros(total_parameters);
        
        if initial_motions.is_some() {
            let value = initial_motions.unwrap();
            assert_eq!(value.len(),number_of_cameras-1); 
            let mut counter = 0;
            for i in (6..number_of_cam_parameters).step_by(12){
                let idx = match i {
                    v if v == 0 => 0,
                    _ => i/6-1-counter
                };
                let (h,rotation_matrix) = value[idx];
                let rotation = na::Rotation3::from_matrix(&rotation_matrix);
                let rotation_transpose = rotation.transpose();
                let translation = rotation_transpose*(-h);
                data.fixed_slice_mut::<3,1>(i,0).copy_from(&translation);
                data.fixed_slice_mut::<3,1>(i,0).copy_from(&rotation_transpose.scaled_axis());
                counter += 1;
            }

        }

        for i in (number_of_cam_parameters..total_parameters).step_by(3){
            data[i] = 0.0;
            data[i+1] = 0.0;
            data[i+2] = initial_depth; 
        }

        State{data , n_cams: number_of_cameras, n_points: number_of_unqiue_points}
    }

    /**
     * This vector has ordering In the format [f1_cam1, f1_cam2,...] where cam_id(cam_n-1) < cam_id(cam_n) 
     */
    pub fn get_observed_features(&self) -> DVector<Float> {
        let n_points = self.number_of_unique_points;
        let n_cams = self.camera_map.keys().len();
        let mut observed_features = DVector::<Float>::zeros(n_points*n_cams*2); // some entries might be zero

        'outer: for r in 0..n_points {
            let row = &self.point_cam_map[r];
            let mut point_found = false;
            for c in 0..n_cams {
                let feat_id = 2*c + 2*r*n_cams;
                let elem = row[c];
                let (x_val, y_val) = match elem {
                    Some(v) => {
                        point_found = true;
                        v
                    },
                    _ => (0.0,0.0)
                };
                observed_features[feat_id] = x_val;
                observed_features[feat_id+1] = y_val;
            }
            if !point_found {
                break 'outer;
            }
        }
        observed_features
    }




}