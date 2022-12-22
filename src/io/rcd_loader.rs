extern crate nalgebra as na;

use std::fs::File;
use std::io::{BufReader,BufRead, Lines};
use na::Matrix3;
use crate::Float;

pub fn load_rotations(file_path: &str) -> Vec<Vec<((usize, usize), Matrix3<Float>)>> {

    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header_raw = lines.next().expect("rcd file empty!").unwrap();
    let header = parse_header(header_raw);

    



    panic!("Not implemented")
}

fn parse_header(header_raw: String) -> Vec<usize> {
    header_raw.split(" ").map(|x| x.trim().parse::<usize>().expect("rcd parsing header val failed")).collect::<Vec<usize>>()
}