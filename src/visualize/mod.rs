extern crate image as image_rs;

use crate::image::Image;
use crate::image::image_encoding::ImageEncoding;
use crate::descriptor::{orientation_histogram::OrientationHistogram,feature_vector::FeatureVector};
use crate::{KeyPoint,Float,float,reconstruct_original_coordiantes};

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

//TODO: maybe account for original x,y
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

pub fn draw_line(image: &mut Image, x_start: usize, y_start: usize, length: Float, angle: Float) -> () {

    let dir_x = length*angle.cos();
    let dir_y = length*angle.sin();

    for i in (0..100).step_by(1) {
        let t = i as Float/100.0;

        let x_pos = (x_start as Float + 0.5 + t*dir_x).trunc() as usize;
        let y_pos = (y_start as Float + 0.5 - t*dir_y).trunc() as usize;

        if x_pos < image.buffer.ncols() && y_pos < image.buffer.nrows()  {
            image.buffer[(y_pos,x_pos)] = 64.0;
        }
    }
    
}

pub fn visualize_keypoint(image: &mut Image, keypoint: &KeyPoint) -> () {
    let radius = (keypoint.octave_level+1) as Float *10.0; 
    assert!(radius > 0.0);
    let (x_orig,y_orig) = reconstruct_original_coordiantes(keypoint.x,keypoint.y,keypoint.octave_level as u32);
    draw_circle(image,x_orig,y_orig, radius);
    draw_line(image, x_orig, y_orig, radius, keypoint.orientation);
}


pub fn draw_square(image: &mut Image, x_center: usize, y_center: usize, side_length: usize) -> () {

    if y_center + side_length >= image.buffer.nrows() || x_center + side_length >= image.buffer.ncols()  {
        println!("Image width,height = {},{}. Max square width,height: {},{}", image.buffer.ncols(), image.buffer.nrows(),x_center+side_length,y_center+side_length);
    } else {
        for i in x_center-side_length..x_center+side_length+1 {
            image.buffer[(y_center + side_length,i)] = 128.0;
            image.buffer[(y_center - side_length,i)] = 128.0;
        }

        for j in y_center-side_length+1..y_center+side_length {
            image.buffer[(j,x_center +side_length)] = 128.0;
            image.buffer[(j,x_center -side_length)] = 128.0;
        }
    }
}

pub fn draw_circle(image: &mut Image, x_center: usize, y_center: usize, radius: Float) -> () {
    for t in (0..360).step_by(1) {
        let rad = (t as Float)*float::consts::PI/180.0;
        let x_pos = (x_center as Float + 0.5 + radius*rad.cos()).trunc() as usize;
        let y_pos = (y_center as Float + 0.5 + radius*rad.sin()).trunc() as usize;

        if x_pos < image.buffer.ncols() && y_pos < image.buffer.nrows()  {
            image.buffer[(y_pos,x_pos)] = 64.0;
        }

    }
}
