extern crate nalgebra as na;
extern crate image as image_rs;

use na::{RowDVector,DMatrix, Vector3, Quaternion};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader,Read,BufRead};

use crate::{Float,float};
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

fn closest_ts_index(ts: Float, list: &Vec<Float>) -> usize {
    let mut min_delta = float::MAX;
    let mut min_idx = list.len()-1;

    for (idx, target_ts) in list.iter().enumerate() {
        let delta = (ts-target_ts).abs();
        
        if delta < min_delta {
            min_delta = delta;
            min_idx = idx;
        } else {
            break;
        }    
    }

    min_idx
}

fn load_timestamps(file_path: &Path)-> Vec<Float> {
    let file = File::open(file_path).expect("load_timestamps failed");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    let mut timestamps = Vec::<Float>::new();

    for line in lines {
        let contents = line.unwrap();
        let values = contents.trim().split(" ").collect::<Vec<&str>>();
        let ts =  parse_to_float(values[0],false);
        timestamps.push(ts);
    }

    timestamps

}



pub fn load_ground_truths(file_path: &Path) -> Vec<(Vector3<Float>,Quaternion<Float>)> {
    let file = File::open(file_path).expect("load_ground_truths failed");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    let mut ground_truths = Vec::<(Vector3<Float>,Quaternion<Float>)>::new();

    for line in lines {
        let contents = line.unwrap();
        if contents.starts_with('#') {
            continue;
        }
        let values = contents.trim().split(" ").map(|x| parse_to_float(x,false)).collect::<Vec<Float>>();
        let tx = values[1];
        let ty = values[2];
        let tz = values[3];

        let qx =  values[4];
        let qy =  values[5];
        let qz =  values[6];
        let qw =  values[7];

        let translation = Vector3::<Float>::new(tx,ty,tz);
        let quaternion = Quaternion::<Float>::new(qw,qx,qy,qz);

        ground_truths.push((translation,quaternion));


    }

    ground_truths
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