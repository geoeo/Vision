extern crate image as image_rs;

use rand::distributions::{Distribution, Uniform};
use crate::features::{Feature, Oriented,orb_feature::OrbFeature, geometry::{point::Point,shape::circle::circle_bresenham,line::line_bresenham}};
use crate::image::{Image,image_encoding::ImageEncoding};
use crate::matching::sift_descriptor::{orientation_histogram::OrientationHistogram};
use crate::{Float,float,reconstruct_original_coordiantes};

pub mod plot;

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

//TODO: Remove OrbFeature dependency or make OrbFeature a more basic feature
pub fn display_matches_for_pyramid<T>(image_a_original: &Image, image_b_original: &Image, match_pyramid: &Vec<((usize,T),(usize,T))>, draw_lines: bool, intensity: Float) -> Image where T: Feature + Oriented {
    let height = image_a_original.buffer.nrows();
    let width = image_a_original.buffer.ncols() + image_b_original.buffer.ncols();

    let mut target_image = Image::empty(width, height, image_b_original.original_encoding);

    for x in 0..image_a_original.buffer.ncols() {
        for y in 0..image_a_original.buffer.nrows() {
            target_image.buffer[(y,x)] = image_a_original.buffer[(y,x)];
        }
    }

    for x in 0..image_b_original.buffer.ncols() {
        for y in 0..image_b_original.buffer.nrows() {
            target_image.buffer[(y,x+image_a_original.buffer.ncols())] = image_b_original.buffer[(y,x)];
        }
    }

    for i in 0..match_pyramid.len() {
        let ((level_a,a),(level_b,b)) = &match_pyramid[i];
            let (a_x_orig,a_y_orig) = reconstruct_original_coordiantes(a.get_x_image(),a.get_y_image(),*level_a as u32);
            let (b_x_orig,b_y_orig) = reconstruct_original_coordiantes(b.get_x_image(),b.get_y_image(),*level_b as u32);
            let match_tuple = (OrbFeature{location: Point::new(a_x_orig, a_y_orig), orientation: a.get_orientation()},
            OrbFeature{location: Point::new(image_a_original.buffer.ncols() + b_x_orig, b_y_orig), orientation: b.get_orientation()} );
            let radius_a = (level_a+1) as Float *10.0; 
            let radius_b = (level_b+1) as Float *10.0; 
    
        draw_match(&mut target_image, &match_tuple, (radius_a,radius_b),draw_lines, intensity);
    }



    target_image

}


//TODO: better drawing solution
fn draw_match<T>(image: &mut Image,  (feature_a,feature_b): &(T,T), (radius_a, radius_b): (Float,Float), draw_lines: bool, intensity: Float)-> ()  where T: Feature + Oriented {

    let intensity_min = intensity/2.0;
    let intensity_max = 3.0*intensity/4.0;

    let range = Uniform::from(intensity_min..intensity_max);
    let mut rng = rand::thread_rng();


    let intenstiy_sample = range.sample(&mut rng);
    draw_circle_with_orientation(image, feature_a.get_x_image(), feature_a.get_y_image(),  feature_a.get_orientation(), radius_a, 0.0);
    draw_circle_with_orientation(image, feature_b.get_x_image(), feature_b.get_y_image(),  feature_b.get_orientation(), radius_b, 0.0);

    if draw_lines {
        let line = line_bresenham(&Point::new(feature_a.get_x_image(), feature_a.get_y_image()), &Point::new(feature_b.get_x_image(), feature_b.get_y_image()));
        draw_points(image, &line.points,0.0);
    }

    

}

fn draw_matches<T>(image: &mut Image,  matches: &Vec<(T,T)>, (radius_a, radius_b): (Float,Float), draw_lines: bool, intensity: Float)-> ()  where T: Feature + Oriented {

    let intensity_min = intensity/2.0;
    let intensity_max = intensity_min + intensity;

    let range = Uniform::from(intensity_min..intensity_max);
    let mut rng = rand::thread_rng();



    for (feature_a,feature_b) in matches {

        let intenstiy_sample = range.sample(&mut rng);
        draw_circle_with_orientation(image, feature_a.get_x_image(), feature_a.get_y_image(),  feature_a.get_orientation(), radius_a, intenstiy_sample);
        draw_circle_with_orientation(image, feature_b.get_x_image(), feature_b.get_y_image(),  feature_b.get_orientation(), radius_b, intenstiy_sample);

        if draw_lines {
            let line = line_bresenham(&Point::new(feature_a.get_x_image(), feature_a.get_y_image()), &Point::new(feature_b.get_x_image(), feature_b.get_y_image()));
            draw_points(image, &line.points,intenstiy_sample);
        }

    }

}

pub fn draw_line(image: &mut Image, x_start: usize, y_start: usize, length: Float, angle: Float, intensity: Float) -> () {

    let dir_x = length*angle.cos();
    //let dir_y = -length*angle.sin();
    let dir_y = length*angle.sin();

    for i in (0..100).step_by(1) {
        let t = i as Float/100.0;

        let x_pos = (x_start as Float + 0.5 + t*dir_x).trunc() as usize;
        let y_pos = (y_start as Float + 0.5 - t*dir_y).trunc() as usize;

        if x_pos < image.buffer.ncols() && y_pos < image.buffer.nrows()  {
            image.buffer[(y_pos,x_pos)] = intensity;
        }
    }
    
}

pub fn visualize_pyramid_feature_with_orientation<T>(image: &mut Image, keypoint: &T, octave_index: usize, intensity: Float) -> () where T: Feature + Oriented {
    let (x_orig,y_orig) = reconstruct_original_coordiantes(keypoint.get_x_image(),keypoint.get_y_image(),octave_index as u32);
    let radius = (octave_index+1) as Float *10.0; 
    draw_circle_with_orientation(image, x_orig, y_orig,  keypoint.get_orientation(), radius, intensity);
}

pub fn draw_circle_with_orientation(image: &mut Image, x: usize, y: usize, orientation : Float, radius: Float, intensity: Float) -> () {
    assert!(radius > 0.0);
    draw_circle(image,x,y, radius, intensity);
    draw_line(image, x, y, radius, orientation,intensity);
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

pub fn draw_circle(image: &mut Image, x_center: usize, y_center: usize, radius: Float, intensity: Float) -> () {
    for t in (0..360).step_by(1) {
        let rad = (t as Float)*float::consts::PI/180.0;
        let x_pos = (x_center as Float + 0.5 + radius*rad.cos()).trunc() as usize;
        let y_pos = (y_center as Float + 0.5 + radius*rad.sin()).trunc() as usize;

        if x_pos < image.buffer.ncols() && y_pos < image.buffer.nrows()  {
            image.buffer[(y_pos,x_pos)] = intensity;
        }

    }
}

pub fn draw_circle_bresenham(image: &mut Image, x_center: usize, y_center: usize, radius: usize) -> () {
    let circle = circle_bresenham(x_center,y_center,radius);
    draw_points(image, &circle.shape.get_points(), 64.0);
}

pub fn draw_points(image: &mut Image, points: &Vec<Point<usize>>, intensity: Float) -> () {
    for point in points {
        if point.x < image.buffer.ncols() && point.y < image.buffer.nrows()  {
            image.buffer[(point.y,point.x)] = intensity;
        }
    }
}


