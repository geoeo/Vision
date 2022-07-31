extern crate nalgebra as na;
extern crate num_traits;
extern crate simba;

use simba::scalar::{SubsetOf,SupersetOf};
use na::{convert, Matrix2,Matrix3,Matrix1x2,Matrix3x1,SMatrix, Vector,SVector,Dim, storage::Storage,DVector, SimdRealField, ComplexField,base::Scalar};
use num_traits::{float,NumAssign, identities};
use crate::image::Image;
use crate::Float;

pub mod lie;
pub mod pose;
pub mod loss;
pub mod least_squares;
pub mod weighting;
pub mod conjugate_gradient;

pub fn to_matrix<F, const N: usize, const M: usize, const D: usize>(vec: &SVector<F,D>) -> SMatrix<F,N,M> where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {
    assert_eq!(D,N*M);

    let mut m = SMatrix::<F,N,M>::zeros();

    for c in 0..M {
        let column = vec.fixed_rows::<N>(c*N);
        m.column_mut(c).copy_from(&column);
    }

    m 
}

pub fn quadratic_roots<F>(a: F, b: F, c: F) -> (F,F) where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {
    let two: F = convert(2.0);
    let four: F = convert(4.0);
    let det = float::Float::powi(b,2)-four*a*c;
    match det {
        det if det > F::zero() => {
            let det_sqrt = float::Float::sqrt(det);
            ((-b - det_sqrt)/two*a,(-b + det_sqrt)/two*a)
        },
        det if det < F::zero() => panic!("determinat less than zero, no real solutions"),
        _ => {
            let res = -b/two*a;
            (res,res)
        }

    }
}

pub fn estimate_std<F>(data: &DVector<F>) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField + identities::One {
    median_absolute_deviation(data)/convert::<f64,F>(0.67449) 
}

pub fn median_absolute_deviation<F>(data: &DVector<F>) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField + identities::One {
    let median_value = median(data.data.as_vec().clone(), true);
    let absolute_deviation: Vec<F> = data.iter().map(|&x| num_traits::float::Float::abs(x-median_value)).collect();
    median(absolute_deviation,false)
}

pub fn median<F>(data: Vec<F>, sort_data: bool) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField + identities::One  {
    let mut mut_data = data;
    if sort_data {
        mut_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let middle = mut_data.len()/2;
    mut_data[middle]
}

pub fn round<F>(number: F, dp: i32) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField{
    let ten: F = convert(10.0);
    let n = float::Float::powi(ten,dp);
    float::Float::round(number * n)/n
}

pub fn calc_sigma_from_z<F>(z: F, x: F, mean: F) -> F  where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {
    assert!(z > F::zero());
    (x-mean)/z
}

pub fn rotation_matrix_2d_from_orientation<F>(orientation: F) -> Matrix2<F> where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {

    Matrix2::new(float::Float::cos(orientation), -float::Float::sin(orientation),
                 float::Float::sin(orientation), float::Float::cos(orientation))

}

pub fn gradient_and_orientation<F>(x_gradient: &Image, y_gradient: &Image, x: usize, y: usize) -> (F,F) where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField + SubsetOf<Float> + SupersetOf<Float> {
    let x_sample: F = convert(x_gradient.buffer.index((y,x)).clone());
    let y_sample: F = convert(y_gradient.buffer.index((y,x)).clone());
    let two: F = convert(2.0);

    let x_diff = round(x_sample,5);
    let y_diff = round(y_sample,5);

    let gradient = float::Float::sqrt(float::Float::powi(x_diff, 2) + float::Float::powi(y_diff,2));
    let orientation = match  y_diff.atan2(x_diff.clone()) {
        angle if angle < F::zero() => two*convert(std::f64::consts::PI) + angle,
        angle => angle
    };

    (gradient,orientation)
}

//TODO: Doesnt seem to work as well as lagrange -> produces out  of scope results
pub fn newton_interpolation_quadratic<F>(a: F, b: F, c: F, f_a: F, f_b: F, f_c: F, range_min: F, range_max: F) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {

    let a_corrected = if a > b { a - range_max} else {a};
    let c_corrected = if b > c { c + range_max} else {c};
    let two: F = convert(2.0);

    assert!( a_corrected < b && b < c_corrected);
    assert!(f_a <= f_b && f_b >= f_c ); 

    let b_2 = (f_b - f_c)/(b-c_corrected);
    let b_3 = (((f_c - f_b)/(c_corrected-b))-((f_b-f_a)/(b-a_corrected)))/(c_corrected-a_corrected);

    let result  = (-b_2 + a_corrected + b) / (two*b_3);

    match result {
        res if res < range_min => res + range_max,
        res if res > range_max => res - range_max,
        res => res
    }

}


// http://fourier.eng.hmc.edu/e176/lectures/NM/node25.html
pub fn lagrange_interpolation_quadratic<F>(a: F, b: F, c: F, f_a: F, f_b: F, f_c: F, range_min: F, range_max: F) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {

    let a_corrected = if a > b { a - range_max} else {a};
    let c_corrected = if b > c { c + range_max} else {c};
    let half: F = convert(0.5);

    assert!( a_corrected < b && b < c_corrected);
    assert!(f_b >= f_a  && f_b >= f_c ); 

    let numerator = (f_a-f_b)*float::Float::powi(c_corrected-b,2)-(f_c-f_b)*float::Float::powi(b-a_corrected,2);
    let denominator = (f_a-f_b)*(c_corrected-b)+(f_c-f_b)*(b-a_corrected);

    let result  = b + half*(numerator/denominator);

    match result {
        res if res < range_min => res + range_max,
        res if res >= range_max => res - range_max,
        res => res
    }
}

// https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
pub fn quadatric_interpolation<F>(a: F, b: F, c: F, f_a: F, f_b: F, f_c: F, range_min: F, range_max: F) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {
    let two: F = convert(2.0);
    let a = Matrix3::new(float::Float::powi(a,2),a,F::one(),
                         float::Float::powi(b,2),b,F::one(),
                         float::Float::powi(c,2),c,F::one());
    let b = Matrix3x1::new(f_a,f_b,f_c);

    let x = a.lu().solve(&b).expect("Linear resolution failed.");

    let coeff_a = x[(0,0)];
    let coeff_b = x[(1,0)];


    let result = -coeff_b/(two*coeff_a);

    match result {
        res if res < range_min => res + range_max,
        res if res >= range_max => res - range_max,
        res => res
    }
}

pub fn gauss_2d<F>(x_center: F, y_center: F, x: F, y: F, sigma: F) -> F where F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField {
    let offset = Matrix1x2::<F>::new(x-x_center,y-y_center);
    let offset_transpose = offset.transpose();
    let sigma_sqr = float::Float::powi(sigma, 2);
    let sigma_sqr_recip = F::one()/sigma_sqr;
    let covariance = Matrix2::<F>::new(sigma_sqr_recip, F::zero(),F::zero(), sigma_sqr_recip);
    let half: F = convert(0.5);
    let two: F = convert(2.0);
    let pi: F = convert(std::f64::consts::PI);

    let exponent = -offset.scale(half)*(covariance*offset_transpose);
    let exp = float::Float::exp(*exponent.index((0,0)));

    let denom = two*pi*sigma_sqr;

    exp/denom
}

pub fn max_norm<F,D,S>(vector: &Vector<F,D,S>) -> F where D: Dim, S: Storage<F,D>, F : num_traits::float::Float + Scalar + NumAssign + SimdRealField + ComplexField  {

    vector.iter().fold(F::zero(),|max,&v| 
        match float::Float::abs(v) {
            v_abs if v_abs > max => v_abs,
            _ => max
        }
    )

}











