extern crate nalgebra as na;

use na::{SMatrix, SVector};
use crate::Float;



/**
  *  m1: Matrix containing the homogeneous coordinates of  feature points in the 1st camera frame.
  *  m2: Matrix containing the homogeneous coordinates of feature points in the 2nd camera frame.
  *  Output: C: The 10*35 coefficient matrix 
  */
pub fn generate_constraints(m1: &SMatrix<Float,3,5>, m2: &SMatrix<Float,3,5>) -> SMatrix<Float,10,35> {

    let num_points = m1.ncols();
    let mut idx_bin_1 = SMatrix::<usize,2,9>::zeros(); // 9 = bionomial_coefficient(num_points,2)-1

    let mut counter = 0;
    for i in 0..(num_points-2) {
        for j in (i+1)..num_points-1 {
            counter+=1;
            idx_bin_1[(0,counter)] = i; 
            idx_bin_1[(1,counter)] = j; 
        }
    }

    let idx_bin_1_row1 =  idx_bin_1.row(0);
    let idx_bin_1_row2 =  idx_bin_1.row(1);

    let mut mx1 = SVector::<Float,9>::zeros();
    let mut mx2 = SVector::<Float,9>::zeros();
    let mut my1 = SVector::<Float,9>::zeros();
    let mut my2 = SVector::<Float,9>::zeros();
    let mut s1 = SVector::<Float,9>::zeros();
    let mut s2 = SVector::<Float,9>::zeros();

    let mut nx1 = SVector::<Float,9>::zeros();
    let mut nx2 = SVector::<Float,9>::zeros();
    let mut ny1 = SVector::<Float,9>::zeros();
    let mut ny2 = SVector::<Float,9>::zeros();
    let mut r1 = SVector::<Float,9>::zeros();
    let mut r2 = SVector::<Float,9>::zeros();

    for i in 0..9 {
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

    //Total number of equations
    // numEq = nchoosek(numPts,3);

    let mut idx_bin_2 = SMatrix::<usize,2,10>::zeros();
    counter = 0;
    let mut counter2 = 0;
    for i in (1..num_points-1).rev(){
        for j in (counter2+1)..(i+counter2){
            for k in (j+1)..(i+counter2+1){
                counter = counter + 1;
                idx_bin_2[(0,counter)] = j;
                idx_bin_2[(1,counter)] = k; 
            }
        }
        counter2 = i + counter2;
    }

    // ai = [num1;
    //      den1];
    let mut a1 = SVector::<Float,20>::zeros();
    let mut a2 = SVector::<Float,20>::zeros();
    let mut a3 = SVector::<Float,20>::zeros();
    let mut a4 = SVector::<Float,20>::zeros();
    let mut a5 = SVector::<Float,20>::zeros();
    let mut a6 = SVector::<Float,20>::zeros();
    let mut a7 = SVector::<Float,20>::zeros();
    let mut a8 = SVector::<Float,20>::zeros();
    let mut a9 = SVector::<Float,20>::zeros();
    let mut a10 = SVector::<Float,20>::zeros();

    let mut b1 = SVector::<Float,20>::zeros();
    let mut b2 = SVector::<Float,20>::zeros();
    let mut b3 = SVector::<Float,20>::zeros();
    let mut b4 = SVector::<Float,20>::zeros();
    let mut b5 = SVector::<Float,20>::zeros();
    let mut b6 = SVector::<Float,20>::zeros();
    let mut b7 = SVector::<Float,20>::zeros();
    let mut b8 = SVector::<Float,20>::zeros();
    let mut b9 = SVector::<Float,20>::zeros();
    let mut b10 = SVector::<Float,20>::zeros();
    for i in 0..10 {
        a1[(0)] = coefs_n[(idx_bin_2[(0,i)],0)];
        a1[(10+i)] = coefs_d[(idx_bin_2[(0,i)],0)];
        a2[(0)] = coefs_n[(idx_bin_2[(0,i)],1)];
        a2[(10+i)] = coefs_d[(idx_bin_2[(0,i)],1)];
        a3[(0)] = coefs_n[(idx_bin_2[(0,i)],2)];
        a3[(10+i)] = coefs_d[(idx_bin_2[(0,i)],2)];
        a4[(0)] = coefs_n[(idx_bin_2[(0,i)],3)];
        a4[(10+i)] = coefs_d[(idx_bin_2[(0,i)],3)];
        a5[(0)] = coefs_n[(idx_bin_2[(0,i)],4)];
        a5[(10+i)] = coefs_d[(idx_bin_2[(0,i)],4)];
        a6[(0)] = coefs_n[(idx_bin_2[(0,i)],5)];
        a6[(10+i)] = coefs_d[(idx_bin_2[(0,i)],5)];
        a7[(0)] = coefs_n[(idx_bin_2[(0,i)],6)];
        a7[(10+i)] = coefs_d[(idx_bin_2[(0,i)],6)];
        a8[(0)] = coefs_n[(idx_bin_2[(0,i)],7)];
        a8[(10+i)] = coefs_d[(idx_bin_2[(0,i)],7)];
        a9[(0)] = coefs_n[(idx_bin_2[(0,i)],8)];
        a9[(10+i)] = coefs_d[(idx_bin_2[(0,i)],8)];
        a10[(0)] = coefs_n[(idx_bin_2[(0,i)],9)];
        a10[(10+i)] = coefs_d[(idx_bin_2[(0,i)],9)]; 
                
        b1[(0)] = coefs_n[(idx_bin_2[(1,i)],0)];
        b1[(10+i)] = coefs_d[(idx_bin_2[(1,i)],0)];
        b2[(0)] = coefs_n[(idx_bin_2[(1,i)],1)];
        b2[(10+i)] = coefs_d[(idx_bin_2[(1,i)],1)];
        b3[(0)] = coefs_n[(idx_bin_2[(1,i)],2)];
        b3[(10+i)] = coefs_d[(idx_bin_2[(1,i)],2)];
        b4[(0)] = coefs_n[(idx_bin_2[(1,i)],3)];
        b4[(10+i)] = coefs_d[(idx_bin_2[(1,i)],3)];
        b5[(0)] = coefs_n[(idx_bin_2[(1,i)],4)];
        b5[(10+i)] = coefs_d[(idx_bin_2[(1,i)],4)];
        b6[(0)] = coefs_n[(idx_bin_2[(1,i)],5)];
        b6[(10+i)] = coefs_d[(idx_bin_2[(1,i)],5)];
        b7[(0)] = coefs_n[(idx_bin_2[(1,i)],6)];
        b7[(10+i)] = coefs_d[(idx_bin_2[(1,i)],6)];
        b8[(0)] = coefs_n[(idx_bin_2[(1,i)],7)];
        b8[(10+i)] = coefs_d[(idx_bin_2[(1,i)],7)];
        b9[(0)] = coefs_n[(idx_bin_2[(1,i)],8)];
        b9[(10+i)] = coefs_d[(idx_bin_2[(1,i)],8)];
        b10[(0)] = coefs_n[(idx_bin_2[(1,i)],9)];
        b10[(10+i)] = coefs_d[(idx_bin_2[(1,i)],9)];  
    }
    
    let coefs_nd = coefs_num_den(&a1,&a2,&a3,&a4,&a5,&a6,&a7,&a8,&a9,&a10,
        &b1,&b2,&b3,&b4,&b5,&b6,&b7,&b8,&b9,&b10);


    //C = coefsND(1:numEq,:) - coefsND(numEq+1:2*numEq,:);

    return coefs_nd.fixed_rows::<10>(0)-coefs_nd.fixed_rows::<10>(10)

}

fn coefs_num(mx1: &SVector::<Float,9>, mx2: &SVector::<Float,9>, my1: &SVector::<Float,9>, my2: &SVector::<Float,9>, nx2: &SVector::<Float,9>, ny2: &SVector::<Float,9>, r2: &SVector::<Float,9>, s1: &SVector::<Float,9>, s2: &SVector::<Float,9>) -> SMatrix<Float,9, 10> {
    let t2 = mx1.component_mul(my2).component_mul(r2);
    let t3 = mx2.component_mul(ny2).component_mul(s1);
    let t4 = my1.component_mul(nx2).component_mul(s2);
    let t5 = mx1.component_mul(nx2).component_mul(s2)*2.0;
    let t6 = my1.component_mul(ny2).component_mul(s2)*2.0;
    let t7 = mx1.component_mul(my2).component_mul(nx2)*2.0;
    let t8 = my2.component_mul(r2).component_mul(s1)*2.0;
    let t9 = mx2.component_mul(my1).component_mul(r2);
    let t10 = mx1.component_mul(ny2).component_mul(s2);
    let t11 = mx2.component_mul(my1).component_mul(ny2)*2.0;
    let t12 = mx2.component_mul(r2).component_mul(s1)*2.0;
    let t13 = my2.component_mul(nx2).component_mul(s1);

    return SMatrix::<Float,9,10>::from_columns(&[
        t2+t3+t4-mx2.component_mul(my1).component_mul(r2)-mx1.component_mul(ny2).component_mul(s2)-my2.component_mul(nx2).component_mul(s1),
        t11+t12-mx1.component_mul(my2).component_mul(ny2)*2.0-mx1.component_mul(r2).component_mul(s2)*2.0,
        t7+t8-mx2.component_mul(my1).component_mul(nx2)*2.0-my1.component_mul(r2).component_mul(s2)*2.0,
        t5+t6-mx2.component_mul(nx2).component_mul(s1)*2.0-my2.component_mul(ny2).component_mul(s1)*2.0,
        -t2-t3+t4+t9+t10-my2.component_mul(nx2).component_mul(s1),
        -t5+t6+mx2.component_mul(nx2).component_mul(s1)*2.0-my2.component_mul(ny2).component_mul(s1)*2.0,
        t7-t8-mx2.component_mul(my1).component_mul(nx2)*2.0+my1.component_mul(r2).component_mul(s2)*2.0,
        -t2+t3-t4+t9-t10+t13,
        -t11+t12+mx1.component_mul(my2).component_mul(ny2)*2.0-mx1.component_mul(r2).component_mul(s2)*2.0,
        t2-t3-t4-t9+t10+t13]
    )
}

fn coefs_dem(mx2: &SVector::<Float,9>, my2: &SVector::<Float,9>, nx1: &SVector::<Float,9>, nx2: &SVector::<Float,9>, ny1: &SVector::<Float,9>, ny2: &SVector::<Float,9>, r1: &SVector::<Float,9>, r2: &SVector::<Float,9>, s2: &SVector::<Float,9>) -> SMatrix<Float,9, 10> {
    let t2 = mx2.component_mul(ny1).component_mul(r2);
    let t3 = my2.component_mul(nx2).component_mul(r1);
    let t4 = nx1.component_mul(ny2).component_mul(s2);
    let t5 = mx2.component_mul(nx2).component_mul(r1)*2.0;
    let t6 = my2.component_mul(ny2).component_mul(r1)*2.0;
    let t7 = mx2.component_mul(nx2).component_mul(ny1)*2.0;
    let t8 = ny1.component_mul(r2).component_mul(s2)*2.0;
    let t9 = my2.component_mul(nx1).component_mul(r2);
    let t10 = nx2.component_mul(ny1).component_mul(s2);
    let t11 = my2.component_mul(nx1).component_mul(ny2)*2.0;
    let t12 = nx1.component_mul(r2).component_mul(s2)*2.0;
    let t13 = mx2.component_mul(ny2).component_mul(r1);

    return SMatrix::<Float,9, 10>::from_columns(&[
        t2+t3+t4-mx2.component_mul(ny2).component_mul(r1)-my2.component_mul(nx1).component_mul(r2)-nx2.component_mul(ny1).component_mul(s2),
        t11+t12-my2.component_mul(nx2).component_mul(ny1)*2.0-nx2.component_mul(r1).component_mul(s2)*2.0,
        t7+t8-mx2.component_mul(nx1).component_mul(ny2)*2.0-ny2.component_mul(r1).component_mul(s2)*2.0,
        t5+t6-mx2.component_mul(nx1).component_mul(r2)*2.0-my2.component_mul(ny1).component_mul(r2)*2.0,
        t2-t3-t4+t9+t10-mx2.component_mul(ny2).component_mul(r1),
        t5-t6-mx2.component_mul(nx1).component_mul(r2)*2.0+my2.component_mul(ny1).component_mul(r2)*2.0,
        -t7+t8+mx2.component_mul(nx1).component_mul(ny2)*2.0-ny2.component_mul(r1).component_mul(s2)*2.0,
        -t2+t3-t4-t9+t10+t13,
        t11-t12-my2.component_mul(nx2).component_mul(ny1)*2.0+nx2.component_mul(r1).component_mul(s2)*2.0,
        -t2-t3+t4+t9-t10+t13
    ]
    )
}

fn coefs_num_den(
    a1: &SVector<Float,20>,a2: &SVector<Float,20>,a3: &SVector<Float,20>,a4: &SVector<Float,20>,a5: &SVector<Float,20>,a6: &SVector<Float,20>,a7: &SVector<Float,20>,a8: &SVector<Float,20>,a9: &SVector<Float,20>,a10: &SVector<Float,20>,
    b1: &SVector<Float,20>,b2: &SVector<Float,20>,b3: &SVector<Float,20>,b4: &SVector<Float,20>,b5: &SVector<Float,20>,b6: &SVector<Float,20>,b7: &SVector<Float,20>,b8: &SVector<Float,20>,b9: &SVector<Float,20>,b10: &SVector<Float,20>
) -> SMatrix<Float,20,35> {
    return SMatrix::<Float,20,35>::from_columns(&[
        a1.component_mul(b1),
        a1.component_mul(b2)+a2.component_mul(b1),
        a2.component_mul(b2)+a1.component_mul(b5)+a5.component_mul(b1),
        a2.component_mul(b5)+a5.component_mul(b2),
        a5.component_mul(b5),
        a1.component_mul(b3)+a3.component_mul(b1),
        a2.component_mul(b3)+a3.component_mul(b2)+a1.component_mul(b6)+a6.component_mul(b1),
        a2.component_mul(b6)+a3.component_mul(b5)+a5.component_mul(b3)+a6.component_mul(b2),
        a5.component_mul(b6)+a6.component_mul(b5),
        a3.component_mul(b3)+a1.component_mul(b8)+a8.component_mul(b1),
        a3.component_mul(b6)+a6.component_mul(b3)+a2.component_mul(b8)+a8.component_mul(b2),
        a6.component_mul(b6)+a5.component_mul(b8)+a8.component_mul(b5),
        a3.component_mul(b8)+a8.component_mul(b3),
        a6.component_mul(b8)+a8.component_mul(b6),
        a8.component_mul(b8),
        a1.component_mul(b4)+a4.component_mul(b1),
        a2.component_mul(b4)+a4.component_mul(b2)+a1.component_mul(b7)+a7.component_mul(b1),
        a2.component_mul(b7)+a4.component_mul(b5)+a5.component_mul(b4)+a7.component_mul(b2),
        a5.component_mul(b7)+a7.component_mul(b5),
        a3.component_mul(b4)+a4.component_mul(b3)+a1.component_mul(b9)+a9.component_mul(b1),
        a3.component_mul(b7)+a4.component_mul(b6)+a6.component_mul(b4)+a7.component_mul(b3)+a2.component_mul(b9)+a9.component_mul(b2),
        a6.component_mul(b7)+a7.component_mul(b6)+a5.component_mul(b9)+a9.component_mul(b5),
        a3.component_mul(b9)+a4.component_mul(b8)+a8.component_mul(b4)+a9.component_mul(b3),
        a6.component_mul(b9)+a7.component_mul(b8)+a8.component_mul(b7)+a9.component_mul(b6),
        a8.component_mul(b9)+a9.component_mul(b8),
        a4.component_mul(b4)+a1.component_mul(b10)+a10.component_mul(b1),
        a4.component_mul(b7)+a7.component_mul(b4)+a2.component_mul(b10)+a10.component_mul(b2),
        a7.component_mul(b7)+a5.component_mul(b10)+a10.component_mul(b5),
        a3.component_mul(b1)+a4.component_mul(b9)+a9.component_mul(b4)+a10.component_mul(b3),
        a6.component_mul(b1)+a7.component_mul(b9)+a9.component_mul(b7)+a10.component_mul(b6),
        a8.component_mul(b1)+a9.component_mul(b9)+a10.component_mul(b8),
        a4.component_mul(b1)+a10.component_mul(b4),
        a7.component_mul(b1)+a10.component_mul(b7),
        a9.component_mul(b1)+a10.component_mul(b9),
        a10.component_mul(b10)
    ])
}




