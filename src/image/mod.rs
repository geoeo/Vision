extern crate image as image_rs;
extern crate nalgebra as na;

use image_rs::{GrayImage, DynamicImage,Pixel, Luma};
use image_rs::flat::NormalForm;
use na::{DMatrix};

use crate::{Float,KeyPoint};
use crate::descriptor::feature_vector::FeatureVector;
use crate::descriptor::orientation_histogram::OrientationHistogram;
use self::image_encoding::ImageEncoding;


pub mod image_encoding;
pub mod filter;
pub mod gauss_kernel;
pub mod prewitt_kernel;
pub mod laplace_kernel;
pub mod kernel;

#[derive(Debug,Clone)]
pub struct Image {
    pub buffer: DMatrix<Float>,
    pub original_encoding: ImageEncoding
}

impl Image {

    pub fn empty(width: usize, height: usize, image_encoding: ImageEncoding) -> Image {
        let buffer =  DMatrix::<Float>::from_element(height,width,0.0);
        Image{ buffer, original_encoding: image_encoding}
    }

    pub fn from_matrix(matrix: &DMatrix<Float>, original_encoding: ImageEncoding, normalize: bool) -> Image {
        let mut buffer = matrix.clone();

        if normalize {
            let max = buffer.amax();
            for elem in buffer.iter_mut() {
                *elem = *elem/max;
            }
        }

        Image{ buffer: buffer, original_encoding}
    }

    pub fn from_gray_image(image: &GrayImage, normalize: bool) -> Image {
        let mut buffer = Image::image_to_matrix(image);

        if normalize {
            let max = buffer.amax();
            for elem in buffer.iter_mut() {
                *elem = *elem/max;
            }
        }

        Image{ buffer: buffer,original_encoding:  ImageEncoding::U8}
    }


    pub fn to_image(&self) -> GrayImage {
        return Image::matrix_to_image(&self.buffer,  self.original_encoding);
    }

    pub fn draw_square(image: &mut Image, x: usize, y: usize, side_length: usize) -> () {

        if y + side_length >= image.buffer.nrows() || x + side_length >= image.buffer.ncols()  {
            println!("Image width,height = {},{}. Max square width,height: {},{}", image.buffer.ncols(), image.buffer.nrows(),x+side_length,y+side_length);
        } else {
            for i in x-side_length..x+side_length+1 {
                image.buffer[(y + side_length,i)] = 128.0;
                image.buffer[(y - side_length,i)] = 128.0;
            }
    
            for j in y-side_length+1..y+side_length {
                image.buffer[(j,x +side_length)] = 128.0;
                image.buffer[(j,x -side_length)] = 128.0;
            }
        }
    }

    pub fn visualize_keypoint(image: &mut Image,x_gradient: &Image, y_gradient: &Image, keypoint: &KeyPoint) -> () {
        let x = keypoint.x;
        let y = keypoint.y;
        let orientation = keypoint.orientation;
        let width = image.buffer.ncols();
        let height = image.buffer.nrows();

        let x_diff = x_gradient.buffer.index((y,x));
        let y_diff = y_gradient.buffer.index((y,x));
        let gradient = (x_diff.powi(2) + y_diff.powi(2)).sqrt();
        let orientation_cos = orientation.cos();
        let orientation_sin = orientation.sin();

        //TODO: maybe optimize this
        for i in (0..100).step_by(1) {
            let t = i as Float/100.0;

            let x_image_end = match x + (t*gradient*orientation_cos).floor() as usize {
                x_pos if x_pos >= image.buffer.ncols() => width -1,
                x_pos => x_pos
            };
            let y_image_end = match y + (t*-gradient*orientation_sin).floor() as usize {
                y_pos if y_pos >= image.buffer.nrows() => height -1,
                y_pos => y_pos
            };

           image.buffer[(y_image_end,x_image_end)] = t*255.0;

        }
        
    }

    pub fn downsample_half(image: &Image) -> Image {
        let width = image.buffer.ncols();
        let height = image.buffer.ncols();

        let new_width = (width as Float)/2.0;
        let new_height = (height as Float)/2.0;

        if new_height.fract() != 0.0 || new_width.fract() != 0.0  {
            panic!("new (height,width): ({},{}) are not multiple of 2",new_height,new_width);
        }

        let mut new_buffer = DMatrix::<Float>::from_element(new_height as usize,new_width as usize,0.0);
        for x in (0..width).step_by(2) {
            for y in (0..height).step_by(2) {
                let new_y = y/2;
                let new_x = x/2;
                new_buffer[(new_y,new_x)] = image.buffer[(y,x)];
            }
        }

        Image{
            buffer: new_buffer,
            original_encoding: image.original_encoding
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

    pub fn display_matches(image_a: &Image, image_b: &Image, features_a: &Vec<FeatureVector>,features_b: &Vec<FeatureVector> , match_indices: &Vec<(usize,usize)>) -> Image {

        assert_eq!(image_a.buffer.nrows(),image_b.buffer.nrows());
        assert_eq!(image_a.buffer.ncols(),image_b.buffer.ncols());

        let height = image_a.buffer.nrows();
        let width = image_a.buffer.ncols() + image_b.buffer.ncols();

        let mut target_image = Image::empty(width, height, image_a.original_encoding);
   
        for x in 0..image_a.buffer.ncols() {
            for y in 0..image_a.buffer.nrows() {
                target_image.buffer[(y,x)] = image_a.buffer[(y,x)];
                target_image.buffer[(y,x+image_a.buffer.ncols())] = image_b.buffer[(y,x)];
            }
        }


        for (a_index,b_index) in match_indices {
            let feature_a = &features_a[a_index.clone()];
            let feature_b = &features_b[b_index.clone()];

            let target_a_x = feature_a.x;
            let target_a_y = feature_a.y;

            let target_b_x = image_a.buffer.ncols() + feature_b.x;
            let target_b_y = feature_b.y;

            Image::draw_square(&mut target_image,target_a_x,target_a_y, 1);
            Image::draw_square(&mut target_image,target_b_x,target_b_y, 1);

            //TODO: Draw line
            
        }

        target_image

    }

    pub fn display_histogram(histogram: &OrientationHistogram, width_scaling:usize, height: usize) -> Image {

        let bin_len = histogram.bins.len();
        let width = width_scaling*bin_len;
        let mut image = Image::empty(width, height, ImageEncoding::U8);
        let mut max_val = histogram.bins[0];
        for i in 1..bin_len {
            let val = histogram.bins[i];
            if val > max_val {
                max_val = val;
            }
        }

        let max_height = height as Float*0.8;

        for i in 0..bin_len {
            let bin_val = histogram.bins[i];
            let scale = bin_val/max_val;
            let bin_height = (max_height*scale) as usize;
            let bin_width = width/bin_len;
            for w in 0..bin_width {
                let x = i*bin_width+w;
                for y in 0..bin_height {
                    image.buffer[(height-1-y,x)] = 255.0;
                }
            }

        }

        image

    }
}



