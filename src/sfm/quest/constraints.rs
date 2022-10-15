extern crate nalgebra as na;
extern crate itertools;

use itertools::Itertools;
use na::{SMatrix,Matrix2xX, DVector, SVector};
use crate::Float;
use crate::numerics::bionomial_coefficient;

/**
  *  m1: Matrix containing the homogeneous coordinates of  feature points in the 1st camera frame.
  *  m2: Matrix containing the homogeneous coordinates of feature points in the 2nd camera frame.
  *  Output: C: The 11*35 coefficient matrix
  */
pub fn generate_constraints(m1: &SMatrix<Float,3,5>, m2: &SMatrix<Float,3,5>) -> SMatrix<Float,11,35> {

    let num_points = m1.ncols();
    let mut idx_bin_1 = Matrix2xX::<usize>::zeros(bionomial_coefficient(num_points,2)-1);

    let mut counter = 0;
    for i in 0..(num_points-3) {
        for j in (i+1)..num_points-1 {
            counter+=1;
            idx_bin_1[(0,counter)] = i; 
            idx_bin_1[(1,counter)] = j; 
        }
    }

    let idx_bin_cols = idx_bin_1.ncols();
    let idx_bin_1_row1 =  idx_bin_1.row(0);
    let idx_bin_1_row2 =  idx_bin_1.row(1);

    let mut mx1 = DVector::<Float>::zeros(idx_bin_cols);
    let mut mx2 = DVector::<Float>::zeros(idx_bin_cols);
    let mut my1 = DVector::<Float>::zeros(idx_bin_cols);
    let mut my2 = DVector::<Float>::zeros(idx_bin_cols);
    let mut s1 = DVector::<Float>::zeros(idx_bin_cols);
    let mut s2 = DVector::<Float>::zeros(idx_bin_cols);

    let mut nx1 = DVector::<Float>::zeros(idx_bin_cols);
    let mut nx2 = DVector::<Float>::zeros(idx_bin_cols);
    let mut ny1 = DVector::<Float>::zeros(idx_bin_cols);
    let mut ny2 = DVector::<Float>::zeros(idx_bin_cols);
    let mut r1 = DVector::<Float>::zeros(idx_bin_cols);
    let mut r2 = DVector::<Float>::zeros(idx_bin_cols);

    for i in 0..idx_bin_cols {
        mx1[i] = m1[(0,idx_bin_1_row1[i])];
        mx2[i] = m1[(0,idx_bin_1_row2[i])];
        nx1[i] = m2[(0,idx_bin_1_row1[i])];
        nx2[i] = m2[(0,idx_bin_1_row2[i])];

        my1[i] = m1[(1,idx_bin_1_row1[i])];
        my2[i] = m1[(1,idx_bin_1_row2[i])];
        ny1[i] = m2[(1,idx_bin_1_row1[i])];
        ny2[i] = m2[(1,idx_bin_1_row2[i])];

        s1[i] = m1[(2,idx_bin_1_row1[i])];
        s2[i] = m1[(2,idx_bin_1_row2[i])];
        r1[i] = m2[(2,idx_bin_1_row1[i])];
        r2[i] = m2[(2,idx_bin_1_row2[i])];
    }

    let coefs_n = coefs_num(&mx1,&mx2,&my1,&my2,&nx2,&ny2,&r2,&s1,&s2);
    let coefs_d = coefs_dem(&mx2,&my2,&nx1,&nx2,&ny1,&ny2,&r1,&r2,&s2);

    panic!("Todo");
}

fn coefs_num(mx1: &DVector::<Float>,mx2: &DVector::<Float>, my1: &DVector::<Float>, my2: &DVector::<Float>, nx2: &DVector::<Float>, ny2: &DVector::<Float>, r2: &DVector::<Float>, s1: &DVector::<Float>, s2: &DVector::<Float>) -> SVector<Float,10> {
    let mut coeffs = SVector::<Float,10>::zeros();

    return coeffs
}

fn coefs_dem(mx2: &DVector::<Float>,my2: &DVector::<Float>, nx1: &DVector::<Float>, nx2: &DVector::<Float>, ny1: &DVector::<Float>, ny2: &DVector::<Float>, r1: &DVector::<Float>, r2: &DVector::<Float>, s2: &DVector::<Float>) -> SVector<Float,10> {
    let mut coeffs = SVector::<Float,10>::zeros();

    return coeffs
}


// t2 = mx1.*my2.*r2;
// t3 = mx2.*ny2.*s1;
// t4 = my1.*nx2.*s2;
// t5 = mx1.*nx2.*s2.*2.0;
// t6 = my1.*ny2.*s2.*2.0;
// t7 = mx1.*my2.*nx2.*2.0;
// t8 = my2.*r2.*s1.*2.0;
// t9 = mx2.*my1.*r2;
// t10 = mx1.*ny2.*s2;
// t11 = mx2.*my1.*ny2.*2.0;
// t12 = mx2.*r2.*s1.*2.0;
// t13 = my2.*nx2.*s1;
// coefsN = [
//     t2+t3+t4-mx2.*my1.*r2-mx1.*ny2.*s2-my2.*nx2.*s1,
// t11+t12-mx1.*my2.*ny2.*2.0-mx1.*r2.*s2.*2.0,
// t7+t8-mx2.*my1.*nx2.*2.0-my1.*r2.*s2.*2.0,
// t5+t6-mx2.*nx2.*s1.*2.0-my2.*ny2.*s1.*2.0,
// -t2-t3+t4+t9+t10-my2.*nx2.*s1,
// -t5+t6+mx2.*nx2.*s1.*2.0-my2.*ny2.*s1.*2.0,
// t7-t8-mx2.*my1.*nx2.*2.0+my1.*r2.*s2.*2.0,
// -t2+t3-t4+t9-t10+t13,
// -t11+t12+mx1.*my2.*ny2.*2.0-mx1.*r2.*s2.*2.0,
// t2-t3-t4-t9+t10+t13];


// end


// %%
// function coefsD = coefsDenVer2_0(mx2,my2,nx1,nx2,ny1,ny2,r1,r2,s2)


// t2 = mx2.*ny1.*r2;
// t3 = my2.*nx2.*r1;
// t4 = nx1.*ny2.*s2;
// t5 = mx2.*nx2.*r1.*2.0;
// t6 = my2.*ny2.*r1.*2.0;
// t7 = mx2.*nx2.*ny1.*2.0;
// t8 = ny1.*r2.*s2.*2.0;
// t9 = my2.*nx1.*r2;
// t10 = nx2.*ny1.*s2;
// t11 = my2.*nx1.*ny2.*2.0;
// t12 = nx1.*r2.*s2.*2.0;
// t13 = mx2.*ny2.*r1;
// coefsD = [t2+t3+t4-mx2.*ny2.*r1-my2.*nx1.*r2-nx2.*ny1.*s2,t11+t12-my2.*nx2.*ny1.*2.0-nx2.*r1.*s2.*2.0,t7+t8-mx2.*nx1.*ny2.*2.0-ny2.*r1.*s2.*2.0,t5+t6-mx2.*nx1.*r2.*2.0-my2.*ny1.*r2.*2.0,t2-t3-t4+t9+t10-mx2.*ny2.*r1,t5-t6-mx2.*nx1.*r2.*2.0+my2.*ny1.*r2.*2.0,-t7+t8+mx2.*nx1.*ny2.*2.0-ny2.*r1.*s2.*2.0,-t2+t3-t4-t9+t10+t13,t11-t12-my2.*nx2.*ny1.*2.0+nx2.*r1.*s2.*2.0,-t2-t3+t4+t9-t10+t13];

// end

