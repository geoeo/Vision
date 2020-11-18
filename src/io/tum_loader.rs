extern crate nalgebra as na;

use na::{RowDVector,DMatrix};
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::Float;

pub fn load_depth_map(file_path: &Path) -> DMatrix<Float> {
    let file = File::open(file_path).expect("load_depth_map failed");
    let reader = BufReader::new(file);
    let lines = reader.lines().collect::<Vec<_>>();
    let rows = lines.len();
    let cols = lines[0].as_ref().unwrap().split(" ").collect::<Vec<&str>>().len();

    let mut matrix = DMatrix::<Float>::zeros(rows,cols);

    for (idx,some_line) in lines.iter().enumerate() {
        let values = some_line.as_ref().unwrap().split(" ").map(|x| x.parse::<Float>().unwrap()).collect::<Vec<Float>>();
        let vector = RowDVector::<Float>::from_vec(values);
        matrix.set_row(idx,&vector);
    }

    matrix
}