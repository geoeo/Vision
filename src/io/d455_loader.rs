extern crate nalgebra as na;
extern crate image as image_rs;

use na::{RowDVector,DMatrix, Vector3, Quaternion};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader,Read,BufRead};

use crate::Float;
use crate::io::{loading_parameters::LoadingParameters,loaded_data::LoadedData, parse_to_float};
use crate::image::{Image,image_encoding::ImageEncoding};
use crate::camera::pinhole::Pinhole;



pub fn load(root_path: &str, parameters: &LoadingParameters) -> LoadedData {
    panic!("not yet implemented");

}