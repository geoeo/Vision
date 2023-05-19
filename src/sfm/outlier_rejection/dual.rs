extern crate nalgebra as na;
extern crate nalgebra_sparse;
extern crate clarabel;

use clarabel::solver::IPSolver;
use num_traits::Zero;
use na::{MatrixXx3,DMatrix,DVector,Isometry3,Matrix,Scalar, Dim, storage::RawStorage};
use std::collections::{HashMap,HashSet};
use std::ops::AddAssign;
use crate::image::features::{Feature};
use crate::Float;

/**
 * Outlier Rejection Using Duality Olsen et al.
 */
pub fn outlier_rejection_dual<Feat: Feature + Clone>(
        camera_ids_root_first: &Vec<usize>, 
        unique_landmark_ids: &mut HashSet<usize>, 
        abs_pose_map: &mut HashMap<usize,Isometry3<Float>>, 
        feature_map: &mut HashMap<usize, Vec<Feat>>,
        tol: Float
    ) -> HashSet<usize> {
    assert_eq!(camera_ids_root_first.len(),abs_pose_map.keys().len());
    assert_eq!(camera_ids_root_first.len(),feature_map.keys().len());
    assert!(unique_landmark_ids.contains(&0)); // ids have to represent matrix indices

    let (a, b, c, a0, b0, c0) = generate_known_rotation_problem(unique_landmark_ids, camera_ids_root_first, abs_pose_map, feature_map);
    let (_, slack) = solve_feasability_problem(a, b, c, a0, b0, c0, tol, 1.0e-1, 100.0);
    compute_receted_landmark_ids(abs_pose_map, feature_map ,unique_landmark_ids, camera_ids_root_first, slack)
}

#[allow(non_snake_case)]
fn solve_feasability_problem(a: DMatrix<Float>, b: DMatrix<Float>, c: DMatrix<Float>, a0: DVector<Float>, b0: DVector<Float>, c0: DVector<Float>, tol: Float, min_depth: Float, max_depth: Float) -> (DVector<Float>, DVector<Float>) {
    let (A,B,C, a_nrows, a1_ncols) = construct_feasability_inputs(a, b, c, a0, b0, c0, tol, min_depth, max_depth);
    let b_rows = B.nrows();
    let (P_cl, A_cl, B_cl, C_cl, cones) = convert_to_clarabel(A,B,C);
    let settings_cl = clarabel::solver::DefaultSettingsBuilder::<Float>::default()
    .equilibrate_enable(true)
    .max_iter(100)
    .build()
    .unwrap();
    let mut solver_cl = clarabel::solver::DefaultSolver::new(&P_cl, &C_cl, &A_cl, &B_cl, &cones, settings_cl);
    solver_cl.solve();

    println!("{:?}", solver_cl.info);
    let Y = DVector::<Float>::from_row_slice(&solver_cl.solution.z[0..b_rows]);

    let mut s_temp = DVector::<Float>::zeros(Y.nrows()-a1_ncols);
    let s_temp_size = s_temp.nrows();
    assert_eq!(s_temp_size,6*a_nrows);

    let mut s = DVector::<Float>::zeros(a_nrows);
    s_temp.rows_mut(0, s_temp.nrows()).copy_from(&Y.rows(a1_ncols,s_temp_size));
    for offset in 0..6 {
        s.add_assign(s_temp.rows(offset*a_nrows,a_nrows));
    }

    let Y_new = Y.rows(0,a1_ncols).into_owned();

    (Y_new, s)
}

#[allow(non_snake_case)]
fn construct_feasability_inputs(a: DMatrix<Float>, b: DMatrix<Float>, c: DMatrix<Float>, a0: DVector<Float>, b0: DVector<Float>, c0: DVector<Float>, tol: Float, min_depth: Float, max_depth: Float) 
    -> (nalgebra_sparse::CscMatrix<Float>, nalgebra_sparse::CscMatrix<Float>, nalgebra_sparse::CscMatrix<Float>, usize, usize) {
    let tol_c = tol*(&c);
    let tol_c0 = tol*(&c0);
    let mut a1 = DMatrix::<Float>::zeros(2*a.nrows() + 2*b.nrows(),a.ncols());
    a1.view_mut((0, 0),a.shape()).copy_from(&(-(&a)-&tol_c));
    a1.view_mut((a.nrows(), 0),a.shape()).copy_from(&((&a)-&tol_c));
    a1.view_mut((2*a.nrows(), 0),b.shape()).copy_from(&(-(&b)-&tol_c));
    a1.view_mut((2*a.nrows() + b.nrows(), 0),b.shape()).copy_from(&((&b)-&tol_c));
    let (a1_nrows,a1_ncols) = a1.shape();
    let a1_csc = to_csc_owned(a1);

    let mut a2 = DMatrix::<Float>::zeros(2*c.nrows(),c.ncols());
    a2.view_mut((0,0), c.shape()).copy_from(&-(&c));
    a2.view_mut((c.nrows(),0), c.shape()).copy_from(&c);
    let (a2_nrows, _) = a2.shape();
    let a2_csc = to_csc_owned(a2);

    let mut A_temp_coo = nalgebra_sparse::CooMatrix::<Float>::new(a1_nrows+a2_nrows,a1_ncols); 
    for (r,c,v) in a1_csc.triplet_iter() {
        A_temp_coo.push(r, c, *v);
    }

    for (r,c,v) in a2_csc.triplet_iter() {
        A_temp_coo.push(r+a1_nrows, c, *v);
    }
    let A_temp_csc = nalgebra_sparse::CscMatrix::from(&A_temp_coo);
    let (A_temp_nrows, A_temp_ncols) = (A_temp_csc.nrows(),A_temp_csc.ncols());

    let mut b1 = DVector::<Float>::zeros(2*a0.nrows() + 2*b0.nrows());
    b1.rows_mut(0, a0.nrows()).copy_from(&(&(a0)+&tol_c0));
    b1.rows_mut(a0.nrows(), a0.nrows()).copy_from(&(-(&a0)+&tol_c0));
    b1.rows_mut(2*a0.nrows(), b0.nrows()).copy_from(&((&b0)+&tol_c0));
    b1.rows_mut(2*a0.nrows() + b0.nrows(), b0.nrows()).copy_from(&(-(&b0)+&tol_c0));
    let b1_csc = to_csc_owned(b1);

    let mut b2 = DVector::<Float>::zeros(2*c0.nrows());
    b2.rows_mut(0,c0.nrows()).copy_from(&(c0.add_scalar(-min_depth)));
    b2.rows_mut(c.nrows(),c0.nrows()).copy_from(&(-c0).add_scalar(max_depth));
    let b2_csc = to_csc_owned(b2);

    let mut C_coo = nalgebra_sparse::CooMatrix::<Float>::new(b1_csc.nrows()+b2_csc.nrows()+A_temp_nrows,1); 
    for (r,c,v) in b1_csc.triplet_iter() {
        C_coo.push(r,c,*v);
    }
    for (r,c,v) in b2_csc.triplet_iter() {
        C_coo.push(r+b1_csc.nrows(),c,*v);
    }
    let C_csc = nalgebra_sparse::CscMatrix::from(&C_coo);

    let mut A_coo = nalgebra_sparse::CooMatrix::new(2*A_temp_nrows,A_temp_ncols + A_temp_nrows);
    for (r,c,v) in A_temp_csc.triplet_iter() {
        A_coo.push(r,c,*v);
    }

    for i in 0..A_temp_nrows {
        A_coo.push(i,i+A_temp_ncols,-1.0);
    }

    for i in 0..A_temp_nrows {
        A_coo.push(i+A_temp_nrows,i+A_temp_ncols,-1.0);
    }

    let mut A_coo_transpose = nalgebra_sparse::CooMatrix::new(A_temp_ncols + A_temp_nrows,2*A_temp_nrows);
    for (r,c,v) in A_coo.triplet_iter() {
        A_coo_transpose.push(c,r,*v);
    }
    let A_transpose_csc = nalgebra_sparse::CscMatrix::from(&A_coo_transpose);


    let mut B = DVector::<Float>::zeros(a1_ncols+A_temp_nrows);
    B.rows_mut(a1_ncols,A_temp_nrows).fill(-1.0); // Implicit Multiplying by -1
    let B_csc = to_csc_owned(B);

    (A_transpose_csc, B_csc, C_csc, a.nrows(), a1_ncols) 
}

fn generate_known_rotation_problem<Feat: Feature + Clone>(unique_landmark_ids: &HashSet<usize>, camera_ids_root_first: &Vec<usize>, abs_pose_map: &mut HashMap<usize,Isometry3<Float>>, feature_map: &HashMap<usize, Vec<Feat>>) -> (DMatrix<Float>,DMatrix<Float>,DMatrix<Float>,DVector<Float>,DVector<Float>,DVector<Float>) {
    let number_of_unique_points = unique_landmark_ids.len();
    let number_of_poses = abs_pose_map.len();
    let number_of_target_parameters = 3*number_of_unique_points + 3*(number_of_poses-1); // The first translation is taken as identity (origin) hence we dont optimize it
    let number_of_residuals = feature_map.values().fold(0, |acc, x| acc + x.len()); //Each observed feature corresponds to one landmark

    let mut a = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut b = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);
    let mut c = DMatrix::<Float>::zeros(number_of_residuals,number_of_target_parameters);

    let a0 = DVector::<Float>::zeros(number_of_residuals);
    let b0 = DVector::<Float>::zeros(number_of_residuals);
    let c0 = DVector::<Float>::zeros(number_of_residuals);

    let mut row_acc = 0;
    // Skip root cam since we assume origin (TOOD: Check this)
    // TODO: check index mapping of camera -> assumes to be consecutive!
    for cam_idx in 1..number_of_poses {
        let cam_id = camera_ids_root_first[cam_idx];
        let rotation = abs_pose_map.get(&cam_id).expect("generate_known_rotation_problem: No rotation found").rotation.to_rotation_matrix();
        let rotation_matrix = rotation.matrix();
        let feature_vec = feature_map.get(&cam_id).expect("generate_known_rotation_problem: No features found");
        let number_of_points = feature_vec.len(); // asuming every feature's landmark id is distinct -> maybe make an explicit check?
        let ones = DVector::<Float>::repeat(number_of_points, 1.0);

        let mut p_data_x = DVector::<Float>::zeros(number_of_points);
        let mut p_data_y = DVector::<Float>::zeros(number_of_points);
        let mut p_col_ids = DVector::<usize>::zeros(number_of_points);

        for p_idx in 0..number_of_points {
            let feature = &feature_vec[p_idx];
            p_data_x[p_idx] = feature.get_x_image_float();
            p_data_y[p_idx] = feature.get_y_image_float();
            p_col_ids[p_idx] = feature.get_landmark_id().expect("generate_known_rotation_problem: no landmark id"); 
        }
        let point_coeff_x = (&p_data_x)*rotation_matrix.row(2) - (&ones)*rotation_matrix.row(0);
        let point_coeff_y = (&p_data_y)*rotation_matrix.row(2) - (&ones)*rotation_matrix.row(1); 
        let point_coeff_z = (&ones)*rotation_matrix.row(2); 

        let mut t_coeff_x = MatrixXx3::<Float>::zeros(number_of_points);
        let mut t_coeff_y = MatrixXx3::<Float>::zeros(number_of_points);
        let mut t_coeff_z = MatrixXx3::<Float>::zeros(number_of_points);
        t_coeff_x.column_mut(0).copy_from(&(-(&ones)));
        t_coeff_y.column_mut(0).copy_from(&(-(&ones)));
        t_coeff_x.column_mut(2).copy_from(&p_data_x);
        t_coeff_y.column_mut(2).copy_from(&p_data_y);
        t_coeff_z.column_mut(2).copy_from(&ones);

        for p_idx in 0..number_of_points{
            let v_p_x = point_coeff_x.row(p_idx);
            let v_p_y = point_coeff_y.row(p_idx);
            let v_p_z = point_coeff_z.row(p_idx);
            let v_t_x = t_coeff_x.row(p_idx);
            let v_t_y = t_coeff_y.row(p_idx);
            let v_t_z = t_coeff_z.row(p_idx);
            let col_id = p_col_ids[p_idx];
            let r_idx = row_acc+p_idx;
            let column_l_idx = 3*col_id;
            let column_t_idx = 3*(number_of_unique_points+(cam_idx-1));
            a.fixed_view_mut::<1,3>(r_idx,column_l_idx).copy_from(&v_p_x);
            a.fixed_view_mut::<1,3>(r_idx,column_t_idx).copy_from(&v_t_x);
            b.fixed_view_mut::<1,3>(r_idx,column_l_idx).copy_from(&v_p_y);
            b.fixed_view_mut::<1,3>(r_idx,column_t_idx).copy_from(&v_t_y);
            c.fixed_view_mut::<1,3>(r_idx,column_l_idx).copy_from(&v_p_z);
            c.fixed_view_mut::<1,3>(r_idx,column_t_idx).copy_from(&v_t_z);
        }

        row_acc += number_of_points; 
    }

    (a,b,c,a0,b0,c0)
}

fn compute_receted_landmark_ids<Feat: Feature + Clone>(
    abs_pose_map: &mut HashMap<usize,Isometry3<Float>>, 
    feature_map: &mut HashMap<usize, Vec<Feat>>, 
    unique_landmark_ids: &mut HashSet<usize>, 
    camera_ids_root_first: &Vec<usize>, 
    slack: DVector<Float>
) -> HashSet<usize> {
    let num_opt_pose = abs_pose_map.len()-1;  // We skipped the root cam since it is the identity pose by definition
    let num_landmarks = unique_landmark_ids.len();
    let mut rejected_landmarks = HashSet::<usize>::with_capacity(num_landmarks*0.1 as usize);

    let mut res_offset = 0;
    for i in 0..num_opt_pose {
        let cam_id = camera_ids_root_first[i+1];

        let feature_vec = feature_map.get(&cam_id).expect("generate_known_rotation_problem: No features found");
        let number_of_landmarks_in_view = feature_vec.len();

        let current_slack = slack.rows(res_offset,number_of_landmarks_in_view);
        for j in 0..feature_vec.len() {
            let s = current_slack[j];
            let f = &feature_vec[j];

            // if s > 1e-7 landmark associated with f is possibly an outlier
            if s > 0.0 {
                println!("Outlier: {}",s);
                // Enable once the whole pipeline is done
                rejected_landmarks.insert(f.get_landmark_id().expect("update_maps: no landmark id"));
            }

        }
        res_offset += number_of_landmarks_in_view;
    }

    rejected_landmarks

}

#[allow(non_snake_case)]
fn convert_to_clarabel(A: nalgebra_sparse::CscMatrix<Float>, B: nalgebra_sparse::CscMatrix<Float>, C: nalgebra_sparse::CscMatrix<Float>) -> (clarabel::algebra::CscMatrix<Float>, clarabel::algebra::CscMatrix<Float>, Vec<Float>, Vec<Float>, [clarabel::solver::SupportedConeT<Float>;2]) {

    let mut B_vec = vec![0.0;B.nrows()];
    for (r,_,v) in B.triplet_iter() {
        B_vec[r] = *v;
    }

    let mut C_vec = vec![0.0;C.nrows()];
    for (r,_,v) in C.triplet_iter() {
        C_vec[r] = *v;
    }

    let B_extension : Vec<Float> = vec![0.0;C.nrows()];
    B_vec.extend(B_extension);

    let mut A_coo_na = nalgebra_sparse::CooMatrix::<Float>::new(A.nrows()+C.nrows(),A.ncols());
    for r in 0..A.nrows() {
        for c in 0..A.ncols() {
            let elem = A.index_entry(r, c);
            match elem {
                nalgebra_sparse::SparseEntry::NonZero(v) => A_coo_na.push(r,c,*v),
                _ => ()
            };
        }
    }

    for i in 0..C.nrows() {
        A_coo_na.push(i+A.nrows(),i,-1.0);
    }

    let A_csc_na = nalgebra_sparse::CscMatrix::<Float>::from(&A_coo_na);

    println!("A_csc_na nnz: {}", A_csc_na.nnz());

    let A_scs_clarabel = clarabel::algebra::CscMatrix::<Float>::new(
        A.nrows()+C.nrows(),
        A.ncols(),
        A_csc_na.col_offsets().to_vec(),
        A_csc_na.row_indices().to_vec(),
        A_csc_na.values().to_vec()
    );

    let cones = [clarabel::solver::ZeroConeT::<Float>(A.nrows()),clarabel::solver::NonnegativeConeT::<Float>(C.nrows())];
    let P_cl = clarabel::algebra::CscMatrix::<Float>::spalloc(C.nrows(), C.nrows(), 0);

    (P_cl, A_scs_clarabel,B_vec,C_vec,cones)
}

fn to_csc_owned<T, R, C, S>(mat: Matrix<T, R, C, S>) -> nalgebra_sparse::CscMatrix<T> where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C> {
    nalgebra_sparse::CscMatrix::<T>::from(&mat)
}