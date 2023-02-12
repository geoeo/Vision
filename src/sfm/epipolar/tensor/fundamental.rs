extern crate nalgebra as na;
extern crate nalgebra_lapack;
extern crate rand;

use std::iter::zip;
use na::{SVector, Vector, Matrix, SMatrix, Matrix3, Vector2, Dynamic, VecStorage, dimension::{U9,U1}, base::storage::Storage};
use nalgebra::linalg::SymmetricEigen;

use crate::Float;
use crate::image::features::{Feature,solver_feature::SolverFeature,Match};
use crate::sfm::epipolar::tensor::Fundamental;

#[allow(non_snake_case)]
pub fn eight_point_least_squares<T : Feature>(matches: &Vec<Match<T>>, f0: Float) -> Fundamental {
    let number_of_matches = matches.len() as Float; 
    assert!(number_of_matches == 8.0);

    let mut M = SMatrix::<Float,9,9>::zeros();

    for m in matches {
        let eta = linear_coefficients(&m.feature_one.get_as_2d_point(), &m.feature_two.get_as_2d_point(), f0);
        M += eta.transpose()*eta;
    }
    let eigen = SymmetricEigen::new(M);

    let mut min_idx = 0;
    let mut min_value  = eigen.eigenvalues[0];
    for i in 1..eigen.eigenvalues.len(){
        if eigen.eigenvalues[i] < min_value {
            min_idx = i;
            min_value = eigen.eigenvalues[i];
        }
    }

    let eigenvector = eigen.eigenvectors.column(min_idx);
    to_fundamental(&eigenvector)

}

/**
 * Photogrammetric Computer Vision p.570
 * Fails if points are coplanar.
 */
#[allow(non_snake_case)]
pub fn eight_point_hartley<T : Feature>(matches: &Vec<Match<T>>, positive_principal_distance: bool, f0: Float) -> Fundamental {
    let number_of_matches = matches.len() as Float; 
    assert!(number_of_matches >= 8.0, "Number of matches: {}", number_of_matches);

    let mut A = Matrix::<Float, Dynamic, U9, VecStorage<Float, Dynamic, U9>>::zeros(matches.len());
    for i in 0..A.nrows() {
        let feature_right = matches[i].feature_two.get_as_2d_point();
        let feature_left = matches[i].feature_one.get_as_2d_point();
        A.row_mut(i).copy_from(&linear_coefficients(&feature_left, &feature_right, f0));
    }

    if A.rank(1e-3) < 8 {
        panic!("Eight Point: Degenerate Feature Configuration!");
    }

    let svd = A.svd(false,true);
    let v_t =  &svd.v_t.expect("SVD failed on A");
    let f = &v_t.row(v_t.nrows()-1);
    
    let F = to_fundamental(&f.transpose());

    let mut svd_f = F.svd(true,true);
    let acc = svd_f.singular_values[0].powi(2) + svd_f.singular_values[1].powi(2);
    svd_f.singular_values[2] = 0.0;
    svd_f.singular_values /= acc.sqrt();
    match positive_principal_distance {
        true => svd_f.recompose().ok().expect("SVD recomposition failed").transpose().normalize(),
        false => svd_f.recompose().ok().expect("SVD recomposition failed").normalize()
    }
    
}

fn to_fundamental<T: Storage<Float,U9,U1>>(f: &Vector<Float, U9, T>) -> Matrix3<Float> {
    Matrix3::<Float>::new(
        f[0],
        f[1],
        f[2],
        f[3],
        f[4],
        f[5],
        f[6],
        f[7],
        f[8]
    )
} 

fn linear_coefficients(feature_left: &Vector2<Float>, feature_right: &Vector2<Float>, f0: Float) -> SMatrix<Float,1, 9> {
    let l_x =  feature_left[0]/f0;
    let l_y =  feature_left[1]/f0;

    let r_x =  feature_right[0]/f0;
    let r_y =  feature_right[1]/f0;

    SMatrix::<Float, 1, 9>::from_vec(vec![
        r_x*l_x,
        r_y*l_x,
        f0*l_x,
        r_x*l_y,
        r_y*l_y,
        f0*l_y,
        f0*r_x,
        f0*r_y,
        f0.powi(2)
    ])
}

/**
 * Compact Fundamental Matrix Computation, Kanatani and Sugaya 
 */
#[allow(non_snake_case)]
pub fn optimal_correction<T : Feature + SolverFeature + Clone>(initial_F: &Fundamental, m_measured_in: &Vec<Match<T>>, f0: Float) -> Fundamental {
    let error_threshold_efns = 1e-4;
    let error_threshold = 1e-4;
    let max_it_efns = 100;
    let max_it = 50;

    let mut m_measured = m_measured_in.clone();
    let mut matches_est = vec![Match { feature_one: T::empty(), feature_two: T::empty(), landmark_id: None }; m_measured.len()];

    let mut it = 0;
    let mut u = linearize_fundamental(initial_F);
    let mut u_cofactor = linear_cofactor(&u);
    let mut F_corrected = initial_F.clone();
    let mut delta = Float::INFINITY;

    while it < max_it && delta > error_threshold  {
        let (u_new_efns, etas, eta_covariances) = EFNS(&m_measured, &matches_est, &u, &u_cofactor, f0, error_threshold_efns, max_it_efns);

        let mut u_new = u_new_efns;
        if u.dot(&u_new) < 0.0 {
            u_new*=-1.0;
        }

        delta = (u - u_new).norm();
        println!("delta: {}",delta);

        F_corrected.copy_from(&to_fundamental(&u));
        
        for i in 0..matches_est.len() {
            let m_est = &mut matches_est[i];
            let m_meas = &mut m_measured[i];
            let eta = &etas[i];
            let eta_cov = &eta_covariances[i];

            let v_one_meas_in = m_measured_in[i].feature_one.get_as_2d_point();
            let v_two_meas_in = m_measured_in[i].feature_two.get_as_2d_point();

            let v_one_meas = m_meas.feature_one.get_as_3d_point(f0);
            let v_two_meas = m_meas.feature_two.get_as_3d_point(f0);
            
            let factor = u_new.dot(eta)/u_new.dot(&(eta_cov*u_new));
            let left_update = factor*SMatrix::<Float,2,3>::from_vec(vec![u_new[0],u_new[3],u_new[1],u_new[4],u_new[2],u_new[5]])*v_one_meas;
            let right_update = factor*SMatrix::<Float,2,3>::from_vec(vec![u_new[0],u_new[1],u_new[3],u_new[4],u_new[6],u_new[7]])*v_two_meas;
            m_est.feature_one.update(&left_update);
            m_est.feature_two.update(&right_update);

            m_meas.feature_one.update(&(v_one_meas_in-left_update));
            m_meas.feature_two.update(&(v_two_meas_in-right_update));
        }

        u.copy_from(&u_new);
        u_cofactor.copy_from(&linear_cofactor(&u));
        it = it+1;
    } 
    
    println!("done");
    F_corrected.normalize()
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

fn compute_eta<T : Feature>(m_measured: &Match<T>, m_est: &Match<T>, f0: Float) -> SVector<Float, 9> {
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
        x_left_measured_sqrd+x_right_measured_sqrd,x_right_measured*y_right_measured,f0*x_right_measured, x_left_measured*y_left_measured,0.0,0.0,f0*x_left_measured,0.0,0.0,
        x_right_measured*y_right_measured,x_left_measured_sqrd+y_right_measured_sqrd,f0*y_right_measured,0.0, x_left_measured*y_left_measured,0.0,0.0,f0*x_left_measured,0.0,
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
fn EFNS<T : Feature>(matches: &Vec<Match<T>>,matches_est: &Vec<Match<T>>, u_orig: &SVector<Float, 9>, u_cofactor: &SVector<Float, 9>, f0: Float, error_threshold: Float, max_it: usize) 
    -> (SVector<Float, 9>, Vec<SVector<Float, 9>>, Vec<SMatrix<Float, 9, 9>>) {

    let mut it = 0;
    let number_of_observations = matches_est.len();
    let mut u_norm = Float::INFINITY;
    let mut u_new = SVector::<Float, 9>::zeros(); 
    let mut u = u_orig.clone();
    //TODO: preallocate an pass in
    let mut etas = Vec::<SVector<Float, 9>>::with_capacity(number_of_observations);
    let mut etas_transposed = Vec::<SMatrix<Float, 1, 9>>::with_capacity(number_of_observations);
    let mut eta_covariances = Vec::<SMatrix<Float, 9, 9>>::with_capacity(number_of_observations);

    while it < max_it && u_norm > error_threshold {
        let mut M =  SMatrix::<Float, 9, 9>::zeros();
        let mut L =  SMatrix::<Float, 9, 9>::zeros();
        let u_transpose = u.transpose();
    
        for (m,m_est) in zip(matches,matches_est) {
            let eta = compute_eta(m, m_est, f0);
            let eta_transpose = eta.transpose();
            let eta_cov = compute_covariance_of_eta(m,f0);

            //println!("{}",eta_cov);

            etas.push(eta);
            etas_transposed.push(eta_transpose);
            eta_covariances.push(eta_cov);
    
            let factor = (u_transpose*eta_cov*u)[0];
            let M_new = eta*eta_transpose /factor;
            let L_new = (u_transpose*eta)[0].powi(2)*eta_cov/factor.powi(2);
    
            M += M_new;
            L += L_new;
        }
    
        let P_cofactor = SMatrix::<Float, 9, 9>::identity() - u_cofactor*u_cofactor.transpose();
        let X = M-L;
        let Y = P_cofactor*X*P_cofactor;
    
        match nalgebra_lapack::Eigen::new(Y, false,true) {
            Some(eigen) => {
                let eigen_vectors = eigen.eigenvectors.expect("EFNS: Eigenvectors Faield!");
                let eigen_values = eigen.eigenvalues_re;
            
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
            
                let v1 = eigen_vectors.column(min_1_idx).normalize();
                let v2 = eigen_vectors.column(min_2_idx).normalize();
            
                let u_hat = (u_transpose*v1)[0]*v1 + (u_transpose*v2)[0]*v2;
                let mut u_prime = (P_cofactor*u_hat).normalize();
        
                if u.dot(&u_prime) < 0.0 {
                    u_prime*=-1.0;
                }
        
                u_norm = (u-u_prime).norm();
                u_new.copy_from(&u_prime);
                u = (u+u_prime).normalize();
        
                println!("EFNS: norm: {} it: {}",u_norm, it);
                    
                it+=1;
            },
            None => it = max_it
        };

    }

    (u_new, etas, eta_covariances)
}

