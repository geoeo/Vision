extern crate image as image_rs;
extern crate nalgebra as na;

use image_rs::{GrayImage, DynamicImage,Pixel, Luma};
use image_rs::flat::NormalForm;
use na::{DMatrix};

use crate::Float;
use self::image_encoding::ImageEncoding;

pub mod image_encoding;

#[derive(Debug,Clone)]
pub struct Image {
    pub buffer: DMatrix<Float>,
    pub original_encoding: ImageEncoding
}

impl Image {
    pub fn from_matrix(matrix: &DMatrix<Float>, original_encoding: ImageEncoding) -> Image {
        let buffer = matrix.clone();
        Image{ buffer: buffer, original_encoding}
    }

    pub fn from_gray_image(image: &GrayImage) -> Image {
        let buffer = image_to_matrix(image);
        Image{ buffer: buffer,original_encoding:  ImageEncoding::U8}
    }


    pub fn to_image(&self) -> GrayImage {
        return matrix_to_image(&self.buffer,  self.original_encoding);
    }
}

fn image_to_matrix(gray_image: &GrayImage) -> DMatrix<Float> {
    debug_assert!(gray_image.sample_layout().is_normal(NormalForm::RowMajorPacked));

    let (width, height) = gray_image.dimensions();
    let size = (width * height) as usize;
    let mut vec_column_major: Vec<Float> = Vec::with_capacity(size);
    for x in 0..width {
        for y in 0..height {
            let pixel = gray_image.get_pixel(x, y);
            let pixel_value = pixel.channels()[0];
            vec_column_major.push(pixel_value as Float);
        }
    }
    DMatrix::<Float>::from_vec(height as usize, width as usize, vec_column_major)
}


fn matrix_to_image(matrix: &DMatrix<Float>,  encoding: ImageEncoding) -> GrayImage {
    let (rows, cols) = matrix.shape();

    let mut gray_image = DynamicImage::new_luma8(cols as u32, rows as u32).to_luma();
    let max = matrix.max();
    let min = matrix.min();
    for c in 0..cols {
        for r in 0..rows {
            let val = *matrix.index((r, c));
            let pixel_value =  encoding.normalize_to_gray(max,min,val);
            gray_image.put_pixel(c as u32, r as u32, Luma([pixel_value]));
        }
    }
    gray_image
}

