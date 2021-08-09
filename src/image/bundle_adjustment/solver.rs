extern crate nalgebra as na;

use na::{DVector,Matrix, Dynamic, U4, VecStorage};
use crate::Float;
use crate::sensors::camera::Camera;
use crate::numerics::lie::exp;
use crate::image::bundle_adjustment::state::State;

/**
 * In the format [f1_cam1,f2_cam1,f3_cam1,f1_cam2,f2_cam2,...]
 * */
pub fn get_estimated_features<C : Camera>(state: &State, cameras: &Vec<&C>, estimated_features: &mut DVector<Float>) -> () {
    let n_cams = state.n_cams;
    let n_points = state.n_points;
    let estimated_state = &state.state;
    assert_eq!(estimated_features.nrows(),n_cams*n_points*2);
    for i in (0..n_cams).step_by(6){
        let u = estimated_state.fixed_rows::<3>(i*n_cams);
        let w = estimated_state.fixed_rows::<3>(i*n_cams+3);
        let pose = exp(&u,&w);
        let camera = cameras[i];

        let mut position_per_cam = Matrix::<Float,U4,Dynamic, VecStorage<Float,U4,Dynamic>>::from_element(n_points, 1.0);
        for j in (0..n_points).step_by(3) {
            position_per_cam.fixed_rows_mut::<3>(j).copy_from(&estimated_state.fixed_rows::<3>(n_cams+j));
        };
        let transformed_points = pose*position_per_cam;
        for j in 0..n_points {
            let estimated_feature = camera.project(&transformed_points.fixed_slice::<3,1>(0,j));            
            estimated_features[i*n_points+2*j] = estimated_feature.x;
            estimated_features[i*n_points+2*j+1] = estimated_feature.y;
        }

    }

}


