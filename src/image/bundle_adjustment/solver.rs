extern crate nalgebra as na;

use na::{DVector,DMatrix,Matrix, Dynamic, U4, VecStorage, Vector4, Matrix2x4};
use crate::Float;
use crate::sensors::camera::Camera;
use crate::numerics::lie::exp;
use crate::image::bundle_adjustment::state::State;

/**
 * In the format [f1_cam1_x,f1_cam1_y,f2_cam1_x,f2_cam1_y,f3_cam1_x,f3_cam1_y,f1_cam2_x,f1_cam2_y,f2_cam2_x,...]
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

pub fn compute_residual(estimated_features: &DVector<Float>, observed_features: &DVector<Float>, residual_vector: &mut DVector<Float>) -> () {
    assert_eq!(residual_vector.nrows(), estimated_features.nrows()/2);
    let diff = observed_features - estimated_features;
    let diff_squared = diff.component_mul(&diff);
    for i in 0..residual_vector.nrows(){
        residual_vector[i] = (diff_squared[2*i] + diff_squared[2*i+1]).sqrt();
    }
}

pub fn compute_jacobian_wrt_object_points<C : Camera>(state: &State, cameras: &Vec<&C>, jacobian: &mut DMatrix<Float>) -> () {
    for i in ((6*state.n_cams)..state.state.ncols()).step_by(3) {
        let point = &state.state.fixed_rows::<3>(i);
        for j in (0..6*state.n_cams).step_by(6) {
            let u = state.state.fixed_rows::<3>(j);
            let w = state.state.fixed_rows::<3>(j+3);
            let transformation = exp(&u,&w);
            let transformed_point = transformation*Vector4::<Float>::new(point[0],point[1],point[2],1.0);
            let projection_jacobian = cameras[j].get_jacobian_with_respect_to_position(&transformed_point.fixed_rows::<3>(0));
            let mut projection_jacobian_homogeneous = Matrix2x4::<Float>::zeros();
            projection_jacobian_homogeneous.fixed_slice_mut::<2,3>(0,0).copy_from(&projection_jacobian);

            let row_idx = ((i/3)*state.n_cams+j)*2;
            let jacobian_homogeneous = &(projection_jacobian_homogeneous*transformation);
            jacobian.fixed_slice_mut::<2,3>(row_idx,6*state.n_cams+i).copy_from(&jacobian_homogeneous.fixed_slice::<2,3>(0,0));
        }

    }

}

pub fn compute_jacobian_wrt_camera_parameters<C : Camera>(state: &State, cameras: &Vec<&C>, jacobian: &mut DMatrix<Float>) -> () {
    panic!("not yet implemented");
}


