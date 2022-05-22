extern crate nalgebra as na;

use na::{DVector,DMatrix};
use std::fs::File;
use std::io::{BufReader,BufRead, Lines};
use crate::io::{parse_to_float};

use crate::Float;

pub fn load_matrix(file_path: &str) -> DMatrix<Float> {

    let file = File::open(file_path).expect("loading octave file failed!");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let (rows,columns, _) = parse_header(&mut lines);

    parse_matrix(&mut lines, rows, columns)
}

pub fn parse_matrix(lines: &mut Lines<BufReader<File>>, rows: usize, columns: usize) ->  DMatrix<Float> {

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

    DMatrix::<Float>::from_row_slice(rows,columns, &vec_data)

}

pub fn parse_header(lines: &mut Lines<BufReader<File>>) -> (usize,usize, bool) {

    let mut rows: usize = 0;
    let mut columns: usize = 0;
    let mut next_matrix_found = false;
    let mut done = false;
    let mut eof = false;

    while !done {
        let line = lines.next();
        if line.is_none() {
            eof = true;
            break;
        }

        let contents = line.expect("Octave file has finished parsing. This should not happen").unwrap();
        if !contents.starts_with('#') {
            rows = 0;
            columns = 0;
            next_matrix_found = false;
            continue;
        }

        let parts = contents.split(":").collect::<Vec<&str>>();
        let type_name = &parts[0][2..];


        match type_name {
            "rows" => rows = parts[1].trim().parse::<usize>().expect("Could not parse row count in octave file!"),
            "type" => {
                match parts[1].trim() {
                    "matrix" => next_matrix_found = true,
                    _ => ()
                }
            },
            "columns" => {
                columns = parts[1].trim().parse::<usize>().expect("Could not parse column count in octave file!");
                // column is always the last entry before data
                if !next_matrix_found {
                    rows = 0;
                    columns = 0;
                }
            },
            _ => ()
        };

        if columns >0 &&  next_matrix_found {
            done = true;
        }
        
    }

    (rows,columns, eof)
}

pub fn load_matrices(file_path: &str) -> Vec<DMatrix<Float>> {

    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut results : Vec<DMatrix<Float>> = Vec::new();
    let mut done = false;

    while !done {
        let (rows,columns, eof) = parse_header(&mut lines);
        match eof {
            true => done = true,
            false => {
                results.push(parse_matrix(&mut lines, rows, columns));
            }
        };
    }

    results 
}

pub fn load_vector(file_path: &str) -> DVector<Float> {

    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let (rows,columns, _) = parse_header(&mut lines);

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