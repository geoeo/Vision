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
    panic!("not yet implemented")
}

pub fn load_timestamps_and_names(file_path: &Path)-> Vec<(Float,String)> {
    panic!("not yet implemented")

}

pub fn load_ground_truths(file_path: &Path) -> Vec<(Vector3<Float>,Quaternion<Float>)> {
    panic!("not yet implemented")
}

pub fn load_depth_image(file_path: &Path, negate_values: bool, invert_y: bool) -> Image {
    panic!("not yet implemented")
}

pub fn load_image_as_gray(file_path: &Path, normalize: bool, invert_y: bool) -> Image {
    panic!("not yet implemented")
}


pub fn load_intrinsics_as_pinhole(file_path: &Path, invert_focal_lengths: bool) -> Pinhole {
    panic!("not yet implemented")
}