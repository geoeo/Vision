extern crate nalgebra as na;

use na::{SMatrix,SVector,Matrix3xX,Matrix4xX,MatrixXx4,OMatrix,RowOVector,U3,U4};
use crate::Float;

#[derive(Clone, Copy)]
pub enum Triangulation {
    LINEAR,
    STEREO
}

//TODO: conditioning, also check what happens to zero entries more thoroughly
/**
 * Linear Triangulartion up to scale. Assuming norm(X) = 1, where X is in homogeneous space.
 * See Triangulation by Hartley et al.
 */
#[allow(non_snake_case)]
pub fn linear_triangulation_svd(image_points_and_projections: &Vec<(&Matrix3xX<Float>, &OMatrix<Float,U3,U4>)>) -> Matrix4xX<Float> {
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