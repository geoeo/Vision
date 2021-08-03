extern crate nalgebra as na;

use na::{DVector,Matrix, Dynamic, U4, VecStorage};
use crate::features::geometry::point::Point;
use crate::Float;
use crate::sensors::camera::Camera;
use crate::numerics::lie::exp;

//TODO: explicit links between camera params and featurs
pub fn get_estimated_features<C : Camera>(estimated_state: &DVector<Float>, features_per_cam: &Vec<usize>, cameras: &Vec<&C>, n_cams: usize, n_features: usize) -> DVector<Float> {
    panic!("not implemented");
    let mut estimated_features = DVector::<Float>::zeros(n_features*2);
    let features_per_cam_acc = features_per_cam.iter().scan(0, |acc, x| {
        *acc = *acc + x;
        Some(*acc)
    }).collect::<Vec<usize>>();
    for i in (0..n_cams).step_by(6){
        let u = estimated_state.fixed_rows::<3>(i*n_cams);
        let w = estimated_state.fixed_rows::<3>(i*n_cams+3);
        let n_features_for_cam = features_per_cam[i];
        let pose = exp(&u,&w);
        let camera = cameras[i];
        let offset = match i {
            0 => n_cams,
            _ => features_per_cam_acc[i-1]
        };

        let position_per_cam = Matrix::<Float,U4,Dynamic, VecStorage<Float,U4,Dynamic>>::from_element(n_features_for_cam, 1.0);
        for j in (offset..features_per_cam_acc[i]).step_by(3) {
            position_per_cam.fixed_rows_mut::<3>(j-offset).copy_from(&estimated_state.fixed_rows::<3>(j));

        };
        let transformed_points = pose*position_per_cam;
        let mut points = Vec::<Point<Float>>::with_capacity(n_features_for_cam);
        for j in 0..n_features_for_cam {
            points.push(camera.project(&transformed_points.fixed_slice::<3,1>(0,j)));
        }

        //TODO:put this in estimated features
    }

    estimated_features

}