extern crate nalgebra as na;

use std::collections::HashMap;
use na::{DVector, Matrix4,SMatrix, SVector,Matrix3x4,Matrix3xX,Matrix4xX,MatrixXx4,OMatrix,RowOVector,U3,U4, Isometry3};
use crate::image::features::{Match,Feature};
use crate::sensors::camera::Camera;
use crate::Float;



#[derive(Clone, Copy)]
pub enum Triangulation {
    LINEAR,
    STEREO
}

pub fn triangulate_matches<Feat: Feature, C: Camera<Float>>(path_pair: (usize, usize), pose_map: &HashMap<(usize, usize), Isometry3<Float>>, 
    match_map: &HashMap<(usize, usize), Vec<Match<Feat>>>, camera_map: &HashMap<usize, C>, triangulation_mode: Triangulation, positive_principal_distance: bool) -> (Matrix4xX<Float>, DVector<Float>){
    let (id1, id2) = path_pair;
    let se3 = pose_map.get(&(id1,id2)).expect(format!("triangulate_matches: pose not found with key: ({},{})",id1,id2).as_str()).to_matrix();
    let ms = match_map.get(&(id1,id2)).expect(format!("triangulate_matches: matches not found with key: ({},{})",id1,id2).as_str());
    let mut normalized_image_points_s = Matrix3xX::<Float>::zeros(ms.len());
    let mut normalized_image_points_f = Matrix3xX::<Float>::zeros(ms.len());

    for i in 0..ms.len() {
        let m = &ms[i];
        let feat_s = m.feature_one.get_as_3d_point(1.0);
        let feat_f = m.feature_two.get_as_3d_point(1.0);
        normalized_image_points_s.column_mut(i).copy_from(&feat_s);
        normalized_image_points_f.column_mut(i).copy_from(&feat_f);
    }

    let f0 = 1.0;
    let f0_prime = 1.0;
    
    let cam_1 = camera_map.get(&id1).expect("triangulate_matches: camera 1 not found");
    let cam_2 = camera_map.get(&id2).expect("triangulate_matches: camera 2 not found");

    let c1_intrinsics = cam_1.get_projection();
    let c2_intrinsics = cam_2.get_projection();
    let transform_c1 = Matrix4::<Float>::identity().fixed_slice::<3,4>(0,0).into_owned();
    let transform_c2 = se3.fixed_slice::<3,4>(0,0).into_owned();
    let projection_1 = c1_intrinsics*transform_c1;
    let projection_2 = c2_intrinsics*transform_c2;

    let landmarks = match triangulation_mode {
        Triangulation::LINEAR => linear_triangulation_svd(&vec!((&normalized_image_points_s,&projection_1),(&normalized_image_points_f,&projection_2)),positive_principal_distance, true),
        Triangulation::STEREO => stereo_triangulation((&normalized_image_points_s,&projection_1),(&normalized_image_points_f,&projection_2),f0,f0_prime).expect("get_euclidean_landmark_state: Stereo Triangulation Failed"),
    };
 
    let reprojection_errors = calculate_reprojection_errors(&landmarks, ms, &transform_c1, cam_1, &transform_c2, cam_2);
    (landmarks, reprojection_errors)
}
 
fn calculate_reprojection_errors<Feat: Feature, C: Camera<Float>>(landmarks: &Matrix4xX<Float>, matches: &Vec<Match<Feat>>, transform_c1: &Matrix3x4<Float>, cam_1 :&C, transform_c2: &Matrix3x4<Float>, cam_2 :&C) -> DVector<Float> {
    let landmark_count = landmarks.ncols();
    let mut reprojection_errors = DVector::<Float>::zeros(landmark_count);

    for i in 0..landmarks.ncols() {
        let p = landmarks.fixed_columns::<1>(i);
        let m = &matches[i];
        let feat_1 = &m.feature_one.get_as_2d_point();
        let feat_2 = &m.feature_two.get_as_2d_point();
        let p_cam_1 = transform_c1*p;
        let p_cam_2 = transform_c2*p;
        let projected_1 = cam_1.project(&p_cam_1).to_vector();
        let projected_2 = cam_2.project(&p_cam_2).to_vector();

        reprojection_errors[i] = (feat_1-projected_1).norm() + (feat_2-projected_2).norm()
    }

    reprojection_errors
}


//TODO: maybe split up sign change
/**
 * Linear Triangulartion up to scale. Assuming norm(X) = 1, where X is in homogeneous space.
 * See Triangulation by Hartley et al.
 */
#[allow(non_snake_case)]
pub fn linear_triangulation_svd(image_points_and_projections: &Vec<(&Matrix3xX<Float>, &OMatrix<Float,U3,U4>)>, positive_principal_distance: bool, flip_points: bool) -> Matrix4xX<Float> {
    let n_cams = image_points_and_projections.len();
    let points_per_cam = image_points_and_projections.first().expect("linear_triangulation: no points!").0.ncols();
    let mut triangulated_points = Matrix4xX::<Float>::zeros(points_per_cam);

    for i in 0..points_per_cam {
        let mut A = MatrixXx4::<Float>::zeros(2*n_cams);
        for j in 0..n_cams {
            let (points, projection) = image_points_and_projections[j];
            let p_1_1 = projection[(0,0)];
            let p_1_2 = projection[(0,1)];
            let p_1_3 = projection[(0,2)];
            let p_1_4 = projection[(0,3)];
    
            let p_2_1 = projection[(1,0)];
            let p_2_2 = projection[(1,1)];
            let p_2_3 = projection[(1,2)];
            let p_2_4 = projection[(1,3)];
    
            let p_3_1 = projection[(2,0)];
            let p_3_2 = projection[(2,1)];
            let p_3_3 = projection[(2,2)];
            let p_3_4 = projection[(2,3)];
            let u = points[(0,i)];
            let v = points[(1,i)];
            A.fixed_rows_mut::<1>(2*j).copy_from(&RowOVector::<Float,U4>::from_vec(vec![u*p_3_1 - p_1_1, u*p_3_2-p_1_2, u*p_3_3-p_1_3, u*p_3_4-p_1_4]));
            A.fixed_rows_mut::<1>(2*j+1).copy_from(&RowOVector::<Float,U4>::from_vec(vec![v*p_3_1 - p_2_1, v*p_3_2-p_2_2, v*p_3_3-p_2_3, v*p_3_4-p_2_4]));
        }


        let svd = (A.transpose()*A).svd(false,true);
        let eigen_vectors = svd.v_t.expect("linear_triangulation: svd failed");

        let p = eigen_vectors.row((2*n_cams)-1);

        triangulated_points[(0,i)] = p[0]/p[3];
        triangulated_points[(1,i)] = p[1]/p[3];
        triangulated_points[(2,i)] = p[2]/p[3];
        triangulated_points[(3,i)] = 1.0;

        if flip_points {
            // We may triangulate points begind the camera. So we flip them depending on the principal distance
            let sign = match (positive_principal_distance,triangulated_points[(2,i)].is_sign_negative()) {
                (false, true) => 1.0,
                (false, false) => -1.0,
                (true, true) => -1.0,
                (true, false) => 1.0
            };

            triangulated_points[(0,i)] *= sign;
            triangulated_points[(1,i)] *= sign;
            triangulated_points[(2,i)] *= sign;
        }


    }
    triangulated_points
}

/**
 * Stereo Triangulation. Only works for two-view problem but optimizes based on the epipolar constraints.
 * See 3D Rotations - Kanatani
 */
#[allow(non_snake_case)]
pub fn stereo_triangulation(image_points_and_projection: (&Matrix3xX<Float>, &OMatrix<Float,U3,U4>), image_points_and_projection_prime: (&Matrix3xX<Float>, &OMatrix<Float,U3,U4>), f0: Float, f0_prime: Float) -> Option<Matrix4xX<Float>> {
    let (image_points, projection) =  image_points_and_projection;
    let (image_points_prime, projection_prime) =  image_points_and_projection_prime;

    assert_eq!(image_points.ncols(),image_points_prime.ncols());
    let n = image_points.ncols();
    let mut triangulated_points = Matrix4xX::<Float>::zeros(n);
    let mut success = true;

    for i in 0..n {
        let im_point = image_points.column(i);
        let im_point_prime = image_points_prime.column(i);
        let x = im_point[(0)];
        let y = im_point[(1)];
        let x_prime = image_points_prime[(0)];
        let y_prime = im_point_prime[(1)];

        let P_11 = projection[(0,0)];
        let P_12 = projection[(0,1)];
        let P_13 = projection[(0,2)];
        let P_14 = projection[(0,3)];

        let P_21 = projection[(1,0)];
        let P_22 = projection[(1,1)];
        let P_23 = projection[(1,2)];
        let P_24 = projection[(1,3)];

        let P_31 = projection[(2,0)];
        let P_32 = projection[(2,1)];
        let P_33 = projection[(2,2)];
        let P_34 = projection[(2,3)];

        let P_11_prime = projection_prime[(0,0)];
        let P_12_prime = projection_prime[(0,1)];
        let P_13_prime = projection_prime[(0,2)];
        let P_14_prime = projection_prime[(0,3)];

        let P_21_prime = projection_prime[(1,0)];
        let P_22_prime = projection_prime[(1,1)];
        let P_23_prime = projection_prime[(1,2)];
        let P_24_prime = projection_prime[(1,3)];

        let P_31_prime = projection_prime[(2,0)];
        let P_32_prime = projection_prime[(2,1)];
        let P_33_prime = projection_prime[(2,2)];
        let P_34_prime = projection_prime[(2,3)];

        let T = SMatrix::<Float,4,3>::new(
            f0*P_11-x*P_31,f0*P_12-x*P_32,f0*P_13-x*P_33,
            f0*P_21-y*P_31,f0*P_22-y*P_32,f0*P_23-y*P_33,
            f0_prime*P_11_prime-x_prime*P_31_prime,f0_prime*P_12_prime-x_prime*P_32_prime,f0_prime*P_13_prime-x_prime*P_33_prime,
            f0_prime*P_21_prime-y_prime*P_31_prime,f0_prime*P_22_prime-y_prime*P_32_prime,f0_prime*P_23_prime-y_prime*P_33_prime
        );
        let p = SVector::<Float,4>::new(
            f0*P_14-x*P_34,
            f0*P_24-y*P_34,
            f0_prime*P_14_prime-x*P_34_prime,
            f0_prime*P_24_prime-y*P_34_prime
        );
        let T_transpose = T.transpose();
        let b = T_transpose*p;

        match (T_transpose*T).lu().solve(&b) {
            Some(x) => {
                triangulated_points[(0,i)] = x[(0,0)];
                triangulated_points[(1,i)] = x[(1,0)];
                triangulated_points[(2,i)] = x[(2,0)];
                triangulated_points[(3,i)] = 1.0;
            },
            _ => {success = false;}
        };


    }

    match success {
        true => Some(triangulated_points),
        false => None
    }
    
}