extern crate nalgebra as na;

use na::{DVector,DMatrix};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader,Read,BufRead};
use crate::io::{parse_to_float};

use crate::Float;

pub fn load_matrix(file_path: &str) -> DMatrix<Float> {

    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();


    let mut count = 1;
    let mut rows: usize = 0;
    let mut columns: usize = 0;
    for _ in 0..5 {
        let line = lines.next();
        let contents = line.unwrap().unwrap();
        match count {
            4 => {
                let vs = contents.split(":").collect::<Vec<&str>>();
                assert!(vs.len() > 1);
                rows = vs[1].trim().parse::<usize>().expect("Could not parse row count in octave file!");
            },
            5 => {
                let vs = contents.split(":").collect::<Vec<&str>>();
                assert!(vs.len() > 1);
                columns = vs[1].trim().parse::<usize>().expect("Could not parse column count in octave file!");
            },
            _ => ()
        }
        count+=1;
    }

    assert!(rows > 0 && columns > 0);

    let mut vec_data: Vec<Float> = Vec::with_capacity(rows*columns);
    for line in lines {
        let row_as_string = line.unwrap();
        if row_as_string.trim().is_empty(){
            break;
        }
        let column_entries_as_string = row_as_string.trim().split(' ').collect::<Vec<&str>>();
        let column_entries = column_entries_as_string.iter().map(|x| parse_to_float(x, false)).collect::<Vec<Float>>();
        vec_data.extend_from_slice(&column_entries);
    }

    DMatrix::<Float>::from_vec(rows,columns, vec_data)
}

pub fn load_vector(file_path: &str) -> DVector<Float> {

    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();


    let mut count = 1;
    let mut rows: usize = 0;
    let mut columns: usize = 0;
    for _ in 0..5 {
        let line = lines.next();
        let contents = line.unwrap().unwrap();
        match count {
            4 => {
                let vs = contents.split(":").collect::<Vec<&str>>();
                assert!(vs.len() > 1);
                rows = vs[1].trim().parse::<usize>().expect("Could not parse row count in octave file!");
            },
            5 => {
                let vs = contents.split(":").collect::<Vec<&str>>();
                assert!(vs.len() > 1);
                columns = vs[1].trim().parse::<usize>().expect("Could not parse column count in octave file!");
            },
            _ => ()
        }
        count+=1;
    }

    assert!(rows > 0 && columns == 1);


    let mut vec_data: Vec<Float> = Vec::with_capacity(rows);
    for line in lines {
        let row_as_string = line.unwrap();
        if row_as_string.trim().is_empty(){
            break;
        }
        let v = parse_to_float(&row_as_string, false);
        vec_data.push(v);
    }

    DVector::<Float>::from_vec( vec_data)
}