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

#[repr(u8)]
pub enum Dataset {
    FR1,
    FR2,
    FR3,
}



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
    let depth_image = image_rs::open(&Path::new(&file_path)).expect("load_image failed").to_luma16();
    Image::from_depth_image(&depth_image,negate_values,invert_y)
}

pub fn load_image_as_gray(file_path: &Path, normalize: bool, invert_y: bool) -> Image {
    let gray_image = image_rs::open(&Path::new(&file_path)).expect("load_image failed").to_luma8();
    Image::from_gray_image(&gray_image,normalize,invert_y)
}


pub fn load_intrinsics_as_pinhole(dataset: &Dataset, invert_focal_lengths: bool) -> Pinhole {
    match dataset {
        Dataset::FR1 => Pinhole::new(517.3, 516.5, 318.6, 255.3, invert_focal_lengths),
        Dataset::FR2 => Pinhole::new(520.9, 521.0, 325.1, 249.7, invert_focal_lengths),
        Dataset::FR3 => Pinhole::new(535.4, 539.2, 320.1, 247.6, invert_focal_lengths)
    }
}