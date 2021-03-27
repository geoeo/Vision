extern crate nalgebra as na;
extern crate image as image_rs;

use std::path::Path;
use std::fs::File;
use std::io::{BufReader,Read};
use na::{RowDVector,DMatrix};
use crate::image::{Image,image_encoding::ImageEncoding};


use crate::{float,Float};

pub mod eth_loader;
pub mod tum_loader;
pub mod d455_loader;
pub mod loading_parameters;


pub fn parse_to_float(string: &str, negate_value: bool) -> Float {
    let parts = string.split("e").collect::<Vec<&str>>();
    let factor = match negate_value {
        true => -1.0,
        false => 1.0
    };

    match parts.len() {
        1 => factor * parts[0].parse::<Float>().unwrap(),
        2 => {
            let num = parts[0].parse::<Float>().unwrap();
            let exponent = parts[1].parse::<i32>().unwrap();
            factor * num*(10f64.powi(exponent) as Float)
        },
        _ => panic!("string malformed for parsing to float: {}", string)
    }
}

pub fn load_depth_image_from_csv(file_path: &Path, negate_values: bool, invert_y: bool, width: usize, height: usize, scale: Float, normalize: bool, set_default_depth: bool) -> Image {
    let file = File::open(file_path).expect("load_depth_map failed");
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).unwrap();

    let mut matrix = DMatrix::<Float>::zeros(height,width);


    let values = contents.trim().split(|c| c == ' ' || c == ',' || c=='\n').map(|x| parse_to_float(x.trim(),negate_values)).collect::<Vec<Float>>();
    let values_scaled = values.iter().map(|&x| x/scale).collect::<Vec<Float>>();
    assert_eq!(values_scaled.len(),height*width);

    for (idx,row) in values_scaled.chunks(width).enumerate() {
        let vector = RowDVector::<Float>::from_row_slice(row);
        let row_idx = match invert_y {
            true => height-1-idx,
            false => idx
        };
        matrix.set_row(row_idx,&vector);
    }

    if set_default_depth {
        fill_matrix_with_default_depth(&mut matrix,negate_values);
    }



    Image::from_matrix(&matrix, ImageEncoding::F64, normalize)
}

pub fn load_image_as_gray(file_path: &Path, normalize: bool, invert_y: bool) -> Image {
    let gray_image = image_rs::open(&Path::new(&file_path)).expect("load_image failed").to_luma8();
    Image::from_gray_image(&gray_image, normalize, invert_y)
}

pub fn load_depth_image(file_path: &Path, negate_values: bool, invert_y: bool, scale: Float, set_default_depth: bool) -> Image {
    let depth_image = image_rs::open(&Path::new(&file_path)).expect("load_image failed").to_luma16();
    let mut image = Image::from_depth_image(&depth_image,negate_values,invert_y);
    image.buffer /= scale;
    if set_default_depth {
        fill_matrix_with_default_depth(&mut image.buffer,negate_values);
    }


    image
}

fn fill_matrix_with_default_depth(target: &mut DMatrix<Float>,negate_values: bool) {
    let extrema = match negate_values {
        true => target.min(),
        false => target.max()
    };

    for r in 0..target.nrows(){
        for c in 0..target.ncols(){
            if target[(r,c)] == 0.0 {
                target[(r,c)] = extrema;
                //target[(r,c)] = 0.0;
            } 
            //TODO: inverse depth
        }
    }

}

// //TODO: can be moved into a more general place for list manipulation
pub fn closest_ts_index(ts: Float, list: &Vec<Float>) -> usize {
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
