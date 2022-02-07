extern crate nalgebra as na;

use na::{DVector,DMatrix};
use std::fs::File;
use std::io::{BufReader,BufRead, Lines};
use crate::io::{load_image_as_gray,parse_to_float, octave_loader::{load_matrices,load_matrix,load_vector}};
use crate::image::Image;

use crate::Float;

pub fn load_images(file_path: &str) -> Vec<Image> {

    let paths = std::fs::read_dir(file_path).unwrap();

    paths.map(|x| x.unwrap().path()).filter(|x| {
        match x.extension() {
            Some(v) => v.to_str().unwrap().ends_with("JPG"),
            _ => false
        }
    }).map(|x| load_image_as_gray(x.as_path(),false,false)).collect()
}