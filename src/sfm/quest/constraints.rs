extern crate nalgebra as na;
extern crate itertools;

use itertools::Itertools;
use na::{SMatrix};
use crate::Float;

pub fn generate_constraints(m1: &SMatrix<Float,3,5>, m2: &SMatrix<Float,3,5>) -> SMatrix<Float,11,35> {

    let num_points = m1.ncols();

    panic!("Todo");
}