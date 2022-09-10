extern crate nalgebra as na;
extern crate nalgebra_lapack;
extern crate rand;

use std::iter::zip;
use na::{SVector, SMatrix, Matrix3,Matrix,Dynamic, VecStorage, dimension::U9};

use crate::Float;
use crate::image::features::{Feature,Match};
use crate::image::epipolar::tensor::Fundamental;

/**
 * Photogrammetric Computer Vision p.570
 * Fails if points are coplanar.
 */
#[allow(non_snake_case)]
pub fn eight_point_hartley<T : Feature>(matches: &Vec<Match<T>>, positive_principal_distance: bool) -> Fundamental {
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

/**
 * Compact Fundamental Matrix Computation, Kanatani and Sugaya 
 */
pub fn optimal_correction(initial_F: &Fundamental) -> Fundamental {
    panic!("TODO")
}

fn linearize_fundamental(f: &Fundamental) -> SVector<Float, 9> {
    SVector::<Float, 9>::from_vec(vec![
        f[(0,0)],
        f[(0,1)],
        f[(0,2)],
        f[(1,0)],
        f[(1,1)],
        f[(1,2)],
        f[(2,0)],
        f[(2,1)],
        f[(2,2)]])
}

fn linear_cofactor(u: &SVector<Float, 9>) ->  SVector<Float, 9> {
    let u1 = u[0];
    let u2 = u[1];
    let u3 = u[2];
    let u4 = u[3];
    let u5 = u[4];
    let u6 = u[5];
    let u7 = u[6];
    let u8 = u[7];
    let u9 = u[8];

    SVector::<Float, 9>::from_vec(vec![
        u5*u9-u8*u6,
        u6*u7-u9*u4,
        u4*u8-u7*u5,
        u8*u3-u2*u9,
        u9*u1-u3*u7,
        u7*u2-u1*u8,
        u2*u6-u5*u3,
        u3*u4-u6*u1,
        u1*u5-u4*u2]).normalize()
}

fn compute_eta<T : Feature>(m_measured: &Match<T>,m_est: &Match<T>, f0: Float) -> SVector<Float, 9> {
    let feature_left_measured = m_measured.feature_one.get_as_2d_point()/f0;
    let feature_right_measured = m_measured.feature_two.get_as_2d_point()/f0;
    let feature_left_est = m_est.feature_one.get_as_2d_point()/f0;
    let feature_right_est = m_est.feature_two.get_as_2d_point()/f0;

    let x_left_measured = feature_left_measured[0];
    let y_left_measured = feature_left_measured[1];
    let x_right_measured = feature_right_measured[0];
    let y_right_measured = feature_right_measured[1];

    let x_left_est = feature_left_est[0];
    let y_left_est = feature_left_est[1];
    let x_right_est = feature_right_est[0];
    let y_right_est = feature_right_est[1];

    SVector::<Float, 9>::from_vec(vec![
        x_left_measured*x_right_measured+x_right_measured*x_left_est+x_left_measured*x_right_est,
        x_left_measured*y_right_measured+y_right_measured*x_left_est+x_left_measured*y_right_est,
        f0*(x_left_measured+x_left_est),
        y_left_measured*x_right_measured+x_right_measured*y_left_est+y_left_measured*x_right_est,
        y_left_measured*y_right_measured+y_right_measured*y_left_est+y_left_measured*y_right_est,
        f0*(y_left_measured+y_left_est),
        f0*(x_right_measured+x_right_est),
        f0*(y_right_measured+y_right_est),
        f0.powi(2)])

}

fn compute_covariance_of_eta<T : Feature>(m_measured: &Match<T>, f0: Float) -> SMatrix<Float, 9, 9> {

    let feature_left_measured = m_measured.feature_one.get_as_2d_point()/f0;
    let feature_right_measured = m_measured.feature_two.get_as_2d_point()/f0;

    let x_left_measured = feature_left_measured[0];
    let y_left_measured = feature_left_measured[1];
    let x_right_measured = feature_right_measured[0];
    let y_right_measured = feature_right_measured[1];

    let x_left_measured_sqrd = x_left_measured.powi(2);
    let y_left_measured_sqrd = y_left_measured.powi(2);
    let x_right_measured_sqrd = x_right_measured.powi(2);
    let y_right_measured_sqrd = y_right_measured.powi(2);

    let f0_sqrd = f0.powi(2);

    SMatrix::<Float, 9, 9>::from_vec(vec! [
        x_left_measured_sqrd+x_right_measured_sqrd,x_right_measured*y_right_measured,f0*x_right_measured,x_left_measured*y_left_measured,0.0,0.0,f0*x_left_measured,0.0,0.0,
        x_right_measured*y_right_measured,x_left_measured_sqrd+y_right_measured_sqrd,f0*y_right_measured,0.0,x_left_measured*y_left_measured,0.0,0.0,f0*x_left_measured,0.0,
        f0*x_right_measured,f0*y_right_measured,f0_sqrd,0.0,0.0,0.0,0.0,0.0,0.0,
        x_left_measured*y_left_measured,0.0,0.0,y_left_measured_sqrd+x_right_measured_sqrd,x_right_measured*y_right_measured,f0*x_right_measured,f0*y_left_measured,0.0,0.0,
        0.0,x_left_measured*y_left_measured,0.0,x_right_measured*y_right_measured,y_left_measured_sqrd+y_right_measured_sqrd,f0*y_right_measured,0.0,f0*y_left_measured,0.0,
        0.0,0.0,0.0,f0*x_right_measured,f0*y_right_measured,f0_sqrd,0.0,0.0,0.0,
        f0*x_left_measured,0.0,0.0,f0*y_left_measured,0.0,0.0,f0_sqrd,0.0,0.0,
        0.0,f0*x_left_measured,0.0,0.0,f0*y_left_measured,0.0,0.0,f0_sqrd,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    ])
}

#[allow(non_snake_case)]
fn EFNS<T : Feature>(matches: &Vec<Match<T>>,matches_est: &Vec<Match<T>>, u_orig: &SVector<Float, 9>, u_cofactor: &SVector<Float, 9>, f0: Float, error_threshold: Float, max_it: usize) -> Option<SVector<Float, 9>> {

    let mut it = 0;
    let mut u_norm = Float::INFINITY;
    let mut u_new: Option<SVector<Float, 9>> = None; 
    let mut u = u_orig.clone();

    while(it < max_it && u_norm > error_threshold) {
        let mut M =  SMatrix::<Float, 9, 9>::zeros();
        let mut L =  SMatrix::<Float, 9, 9>::zeros();
        let u_transpose = u.transpose();
    
        for (m,m_est) in zip(matches,matches_est) {
            let eta = compute_eta(m, m_est, f0);
            let eta_transpose = eta.transpose();
            let eta_cov = compute_covariance_of_eta(m,f0);
    
            let factor = (u_transpose*eta_cov*u)[0];
            let M_new = eta*eta_transpose /factor;
            let L_new = (u_transpose*eta)[0].powi(2)*eta_cov/factor.powi(2);
    
            M += M_new;
            L += L_new;
        }
    
        let P_cofactor = SMatrix::<Float, 9, 9>::identity() - u_cofactor*u_cofactor.transpose();
        let X = M-L;
        let Y = P_cofactor*X*P_cofactor;
    
        let eigen = nalgebra_lapack::Eigen::new(Y, false,true).expect("EFNS: Eigen Decomp Faield!");
        let eigen_vectors = eigen.eigenvectors.expect("EFNS: Eigenvectors Faield!");
        let eigen_values = eigen.eigenvalues;
    
        let mut min_1_idx = 0;
        let mut min_1_val = Float::INFINITY;
        let mut min_2_idx = 0;
        let mut min_2_val = Float::INFINITY;
    
        for i in 0..9 {
            match eigen_values[i].abs() {
                v if v < min_1_val && v < min_2_val => {
                    min_2_idx = min_1_idx;
                    min_2_val = min_1_val;
                    min_1_idx = i;
                    min_1_val = v;
                },
                v if v > min_1_val && v < min_2_val => {
                    min_2_idx = i;
                    min_2_val = v;
                },
                _ => ()
            };
        }
    
        let v1 = eigen_vectors.column(min_1_idx);
        let v2 = eigen_vectors.column(min_2_idx);
    
        let u_hat = (u_transpose*v1)[0]*v1 + (u_transpose*v2)[0]*v2;
        let u_prime = (P_cofactor*u_hat).normalize();
        u_norm = (u-u_prime).norm();

        match u_norm {
            v if v < error_threshold => u_new = Some(u_prime),
            _ => u = (u+u_prime).normalize()
        };
            

        it+=1;
    }

    u_new


}

