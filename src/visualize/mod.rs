extern crate image as image_rs;
extern crate nalgebra as na;

use rand::distributions::{Distribution, Uniform};
use na::Vector3;
use crate::image::features::{ImageFeature,Feature,Match, Oriented,orb_feature::OrbFeature, geometry::{point::Point,shape::circle::circle_bresenham,line::line_bresenham}};
use crate::image::{Image,image_encoding::ImageEncoding};
use crate::image::descriptors::sift_descriptor::{orientation_histogram::OrientationHistogram};
use crate::{Float,float,reconstruct_original_coordiantes_for_float};

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

pub fn display_oriented_matches_for_pyramid<T: Feature + Oriented>(image_a_original: &Image, image_b_original: &Image, match_pyramid: &Vec<Match<T>>, draw_lines: bool, intensity: Float, pyramid_scale: Float) -> Image {
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
        let a= &match_pyramid[i].feature_one;
        let b = &match_pyramid[i].feature_two;
        let level_a = a.get_closest_sigma_level();
        let level_b = b.get_closest_sigma_level();
        let (a_x_orig,a_y_orig) = reconstruct_original_coordiantes_for_float(a.get_x_image() as Float,a.get_y_image() as Float, pyramid_scale,level_a as i32);
        let (b_x_orig,b_y_orig) = reconstruct_original_coordiantes_for_float(b.get_x_image() as Float,b.get_y_image() as Float, pyramid_scale,level_b as i32);
        let match_tuple = (OrbFeature{location: Point::new(a_x_orig.trunc() as usize, a_y_orig.trunc() as usize), orientation: a.get_orientation(), sigma_level: i},
        OrbFeature{location: Point::new(image_a_original.buffer.ncols() + (b_x_orig.trunc() as usize), b_y_orig.trunc() as usize), orientation: b.get_orientation(), sigma_level: i} );
        let radius_a = (level_a+1) as Float *10.0; 
        let radius_b = (level_b+1) as Float *10.0; 
    
        draw_match_points_with_orientation(&mut target_image, &match_tuple, (radius_a,radius_b),draw_lines, intensity);
    }

    target_image
}

pub fn display_matches_for_pyramid<T: Feature>(image_a_original: &Image, image_b_original: &Image, match_pyramid: &Vec<Match<T>>, draw_lines: bool, intensity: Float, pyramid_scale: Float, invert_y: bool) -> Image {
    let height = image_a_original.buffer.nrows();
    let height_f = height as Float;
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
        let a = &match_pyramid[i].feature_one;
        let b = &match_pyramid[i].feature_two;
        let level_a = a.get_closest_sigma_level();
        let level_b = b.get_closest_sigma_level();
        let (a_y, b_y) = match invert_y {
            true => (height_f - 1.0 - a.get_y_image_float(), height_f - 1.0 - b.get_y_image_float()),
            false => (a.get_y_image_float(), b.get_y_image_float())
        };
        let (a_x_orig, a_y_orig) = reconstruct_original_coordiantes_for_float(a.get_x_image() as Float, a_y, pyramid_scale,level_a as i32);
        let (b_x_orig, b_y_orig) = reconstruct_original_coordiantes_for_float(b.get_x_image() as Float, b_y as Float, pyramid_scale,level_b as i32);
        let match_tuple = (ImageFeature{location: Point::new(a_x_orig.trunc() , a_y_orig.trunc())},
        ImageFeature{location: Point::new((image_a_original.buffer.ncols() + (b_x_orig.trunc() as usize)) as Float, b_y_orig.trunc())} );
        let radius_a = (level_a+1) as Float *10.0; 
        let radius_b = (level_b+1) as Float *10.0; 
    
        draw_match_points(&mut target_image, &match_tuple, (radius_a,radius_b),draw_lines, intensity);
    }

    target_image
}

fn draw_match_points<T: Feature>(image: &mut Image,  (feature_a, feature_b): &(T,T), (radius_a, radius_b): (Float,Float), draw_lines: bool, intensity: Float)-> () {
    draw_circle(image, feature_a.get_x_image(), feature_a.get_y_image(), radius_a, intensity);
    draw_circle(image, feature_b.get_x_image(), feature_b.get_y_image(), radius_b, intensity);

    if draw_lines {
        let line = line_bresenham(&Point::new(feature_a.get_x_image(), feature_a.get_y_image()), &Point::new(feature_b.get_x_image(), feature_b.get_y_image()));
        draw_points(image, &line.points,intensity);
    }
}

fn draw_match_points_with_orientation<T: Feature + Oriented>(image: &mut Image,  (feature_a,feature_b): &(T,T), (radius_a, radius_b): (Float,Float), draw_lines: bool, intensity: Float)-> () {
    draw_circle_with_orientation(image, feature_a.get_x_image(), feature_a.get_y_image(),  feature_a.get_orientation(), radius_a, intensity);
    draw_circle_with_orientation(image, feature_b.get_x_image(), feature_b.get_y_image(),  feature_b.get_orientation(), radius_b, intensity);

    if draw_lines {
        let line = line_bresenham(&Point::new(feature_a.get_x_image(), feature_a.get_y_image()), &Point::new(feature_b.get_x_image(), feature_b.get_y_image()));
        draw_points(image, &line.points,intensity);
    }
}

fn draw_matches<T: Feature + Oriented>(image: &mut Image,  matches: &Vec<(T,T)>, (radius_a, radius_b): (Float,Float), draw_lines: bool, intensity: Float)-> () {

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

pub fn visualize_pyramid_feature_with_orientation<T: Feature + Oriented>(image: &mut Image, keypoint: &T, octave_index: usize, pyrmaid_scale: Float, intensity: Float) -> () {
    let (x_orig,y_orig) = reconstruct_original_coordiantes_for_float(keypoint.get_x_image() as Float,keypoint.get_y_image() as Float, pyrmaid_scale ,octave_index as i32);
    let radius = (octave_index+1) as Float *10.0; 
    draw_circle_with_orientation(image, x_orig.trunc() as usize, y_orig.trunc() as usize,  keypoint.get_orientation(), radius, intensity);
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

pub fn draw_epipolar_lines(image_from: &mut Image, image_to: &mut Image, line_intensity: Float , epipolar_lines: &Vec<(Vector3<Float>, Vector3<Float>)>) -> () {
    let width_from = image_from.buffer.ncols();
    let height_from = image_from.buffer.nrows();

    let width_to = image_to.buffer.ncols();
    let height_to = image_to.buffer.nrows();

    for (l_from, l_to) in epipolar_lines  {

        let x_from_start = 0;
        let y_from_start = match (-l_from[2]/l_from[1]).floor() as usize {
            v if v > height_from => height_from-1,
            v => v
        };

        let x_from_end = width_from-1;
        let y_from_end = match (-(l_from[2] + (x_from_end as Float)*l_from[0])/l_from[1]).floor() as usize {
            v if v > height_from => height_from-1,
            v => v
        };

        let x_to_start = 0;
        let y_to_start = match (-l_to[2]/l_to[1]).floor() as usize {
            v if v > height_to => height_to-1,
            v =>  v
        };

        let x_to_end = width_to-1;
        let y_to_end = match (-(l_to[2] + (x_to_end as Float)*l_to[0])/l_to[1]).floor() as usize {
            v if v > height_to => height_to-1,
            v => v
        };

        let line_from = line_bresenham(&Point::new(x_from_start, y_from_start), &Point::new(x_from_end,y_from_end));
        draw_points(image_to, &line_from.points, line_intensity);

        let line_to = line_bresenham(&Point::new(x_to_start, y_to_start), &Point::new(x_to_end,y_to_end));
        draw_points(image_from, &line_to.points, line_intensity);
    }

}


