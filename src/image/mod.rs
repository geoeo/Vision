extern crate image as image_rs;
extern crate nalgebra as na;

use image_rs::{GrayImage,ImageBuffer, DynamicImage,Pixel, Luma};
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

    pub fn size(&self) -> usize {
        self.buffer.ncols()*self.buffer.nrows()
    }

    pub fn empty(width: usize, height: usize, image_encoding: ImageEncoding) -> Image {
        let buffer =  DMatrix::<Float>::from_element(height,width,255.0);
        Image{ buffer, original_encoding: image_encoding}
    }

    pub fn from_matrix(matrix: &DMatrix<Float>, original_encoding: ImageEncoding, normalize: bool) -> Image {
        let mut buffer = matrix.clone();

        //TODO use nalgebra
        if normalize {
            let max = buffer.amax();
            for elem in buffer.iter_mut() {
                *elem = *elem/max;
            }
        }

        Image{ buffer: buffer, original_encoding}
    }

    pub fn from_gray_image(image: &GrayImage , normalize: bool, invert_y : bool) -> Image {
        let mut buffer = Image::image8_to_matrix(image, invert_y);
        
        //TODO use nalgebra
        if normalize {
            let max = buffer.amax();
            for elem in buffer.iter_mut() {
                *elem = *elem/max;
            }
        }

        Image{ buffer: buffer,original_encoding:  ImageEncoding::U8}
    }

    pub fn from_depth_image(image: &ImageBuffer<Luma<u16>, Vec<u16>> ,  negate_values: bool, invert_y : bool) -> Image {
        let mut buffer = Image::image16_to_matrix(image, invert_y);
        
        if negate_values {
            buffer *= -1.0;
        }

        Image{ buffer: buffer,original_encoding:  ImageEncoding::U16}
    }


    pub fn to_image(&self) -> GrayImage {
        return Image::matrix_to_image(&self.buffer,  self.original_encoding);
    }

    // pub fn to_image16(&self) -> GrayImage {
    //     return Image::matrix_to_image16(&self.buffer,  self.original_encoding);
    // }

    pub fn normalize(&self) -> Image {
        Image{ buffer: self.buffer.normalize(), original_encoding:  self.original_encoding}
    }

    pub fn center(&self) -> Image {
        let mean = self.buffer.mean();
        Image{ buffer: (self.buffer.add_scalar(-mean)), original_encoding:  self.original_encoding}

    }

    pub fn z_standardize(&self) -> Image {
        let mean = self.buffer.mean();
        let variance = self.buffer.variance();
        let std_dev = variance.sqrt();
        Image{ buffer: (self.buffer.add_scalar(-mean))/std_dev, original_encoding:  self.original_encoding}

    }


    pub fn downsample_half(image: &Image, normalize: bool, (r_min,c_min): (usize,usize)) -> Image {
        let width = image.buffer.ncols();
        let height = image.buffer.nrows();

        let new_width = ((width as Float)/2.0).trunc() as usize;
        let new_height = ((height as Float)/2.0).trunc() as usize;

        if new_height < r_min || new_width < c_min  {
            panic!("new (height,width): ({},{}) are too small",new_height,new_width);
        }

        let mut new_buffer = DMatrix::<Float>::from_element(new_height,new_width,0.0);
        for x in (0..width).step_by(2) {
            for y in (0..height).step_by(2) {
                let new_y = y/2;
                let new_x = x/2;
                if new_y < new_height && new_x < new_width {
                    new_buffer[(new_y,new_x)] = image.buffer[(y,x)];
                }

            }
        }

        if normalize {
            new_buffer.normalize_mut();
        }

        Image{
            buffer: new_buffer,
            original_encoding: image.original_encoding
        }

    }

    pub fn upsample_double(image: &Image, normalize: bool) -> Image {
        let width = image.buffer.ncols();
        let height = image.buffer.nrows();

        let new_width = width*2;
        let new_height = height*2;

        let old_buffer = &image.buffer;
        let mut new_buffer = DMatrix::<Float>::zeros(new_height,new_width);

        for x in 0..new_width-2 {
            for y in 0..new_height-2 {
                let x_prime = x as Float / 2.0;
                let y_prime = y as Float / 2.0;
                let x_prime_trunc = x_prime.trunc();
                let y_prime_trunc = y_prime.trunc();

                let new_val = (x_prime - x_prime_trunc)*(y_prime - y_prime_trunc)*old_buffer[(y_prime_trunc as usize + 1, x_prime_trunc as usize + 1)] + 
                              (1.0 + x_prime_trunc - x_prime)*(y_prime - y_prime_trunc)*old_buffer[(y_prime_trunc as usize + 1, x_prime_trunc as usize)] +
                              (x_prime - x_prime_trunc)*(1.0 + y_prime_trunc - y_prime)*old_buffer[(y_prime_trunc as usize, x_prime_trunc as usize + 1)] + 
                              (1.0 + x_prime_trunc - x_prime)*(1.0 + y_prime_trunc - y_prime)*old_buffer[(y_prime_trunc as usize, x_prime_trunc as usize)];

                              
                              
                new_buffer[(y,x)] = new_val;
            }
        }

        if normalize {
            new_buffer.normalize_mut();
        }

        Image{
            buffer: new_buffer,
            original_encoding: image.original_encoding
        }

    }

    fn image8_to_matrix(gray_image: &GrayImage, invert_y: bool) -> DMatrix<Float> {
        debug_assert!(gray_image.sample_layout().is_normal(NormalForm::RowMajorPacked));
    
        let (width, height) = gray_image.dimensions();
        let size = (width * height) as usize;
        let mut vec_column_major: Vec<Float> = Vec::with_capacity(size);
        for x in 0..width {
            for y in 0..height {
                let pixel = match invert_y {
                    true =>  gray_image.get_pixel(x, height - 1 - y),
                    false => gray_image.get_pixel(x, y)
                };
                let pixel_value = pixel.channels()[0];
                vec_column_major.push(pixel_value as Float);
            }
        }
        DMatrix::<Float>::from_vec(height as usize, width as usize, vec_column_major)
    }

    fn image16_to_matrix(gray_image: &ImageBuffer<Luma<u16>, Vec<u16>>, invert_y: bool) -> DMatrix<Float> {
        debug_assert!(gray_image.sample_layout().is_normal(NormalForm::RowMajorPacked));
    
        let (width, height) = gray_image.dimensions();
        let size = (width * height) as usize;
        let mut vec_column_major: Vec<Float> = Vec::with_capacity(size);
        for x in 0..width {
            for y in 0..height {
                let pixel = match invert_y {
                    true =>  gray_image.get_pixel(x, height - 1 - y),
                    false => gray_image.get_pixel(x, y)
                };
                let pixel_value = pixel.channels()[0];
                vec_column_major.push(pixel_value as Float);
            }
        }
        DMatrix::<Float>::from_vec(height as usize, width as usize, vec_column_major)
    }
    
    
    fn matrix_to_image(matrix: &DMatrix<Float>,  encoding: ImageEncoding) -> GrayImage {
        let (rows, cols) = matrix.shape();
    
        let mut gray_image = DynamicImage::new_luma8(cols as u32, rows as u32).to_luma8();
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

    //TODO: normalize flag
    // fn matrix_to_image16(matrix: &DMatrix<Float>,  encoding: ImageEncoding) -> ImageBuffer<Luma<u16>, Vec<u16>> {
    //     let (rows, cols) = matrix.shape();
    
    //     let mut gray_image = DynamicImage::new_luma16(cols as u32, rows as u32).to_luma16();
    //     let max = matrix.max();
    //     let min = matrix.min();
    //     for c in 0..cols {
    //         for r in 0..rows {
    //             let val = *matrix.index((r, c));
    //             let pixel_value =  encoding.normalize_to_gray(max,min,val);
    //             gray_image.put_pixel(c as u32, r as u32, Luma([pixel_value]));
    //         }
    //     }
    //     gray_image
    // }





}



