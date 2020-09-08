extern crate image as image_rs;

use crate::image::Image;
use crate::image::image_encoding::ImageEncoding;
use crate::descriptor::{orientation_histogram::OrientationHistogram,feature_vector::FeatureVector};
use crate::{KeyPoint,Float};

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

        draw_square(&mut target_image,target_a_x,target_a_y, 1);
        draw_square(&mut target_image,target_b_x,target_b_y, 1);

        //TODO: Draw line
        
    }

    target_image

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