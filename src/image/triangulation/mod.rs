extern crate nalgebra as na;

use na::{Matrix3xX,Matrix4xX,MatrixXx4,OMatrix,RowOVector,U3,U4};
use crate::Float;

//TODO: conditioning, also check what happens to zero entries more thoroughly
/**
 * Linear Triangulartion up to scale. Assuming norm(X) = 1, where X is in homogeneous space.
 * See Triangulation by Hartley et al.
 */
#[allow(non_snake_case)]
pub fn linear_triangulation(image_points_and_projections: &Vec<(&Matrix3xX<Float>, &OMatrix<Float,U3,U4>)>) -> Matrix4xX<Float> {
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
        let svd = A.svd(false,true);
        let eigen_vectors = svd.v_t.expect("linear_triangulation: svd failed");

        let p = eigen_vectors.row((2*n_cams)-1);

        triangulated_points[(0,i)] = p[0]/p[3];
        triangulated_points[(1,i)] = p[1]/p[3];
        triangulated_points[(2,i)] = p[2]/p[3];
        triangulated_points[(3,i)] = 1.0;
    }
    triangulated_points
}