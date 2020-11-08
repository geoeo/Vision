extern crate rand_distr;
extern crate rand;
extern crate nalgebra as na;

use rand::prelude::ThreadRng;
use rand_distr::{Normal,Distribution};
use na::DMatrix;

use crate::image::Image;
use crate::features::{geometry::point::Point, orb_feature::OrbFeature};
use crate::Float;
use crate::numerics::rotation_matrix_2d_from_orientation;
use self::bit_vector::BitVector;


pub mod bit_vector;

#[derive(Debug,Clone)]
pub struct BriefDescriptor {
    bit_vector: BitVector
}

impl BriefDescriptor {

    pub fn match_descriptors(descriptors_a: &Vec<&BriefDescriptor>, descriptors_b: &Vec<&BriefDescriptor>) -> Vec<usize> {
        descriptors_a.iter().map(|x| BriefDescriptor::best_match_against(x, descriptors_b)).collect::<Vec<usize>>()
    }

    //TODO: if min distance is above a thrshold then return None
    pub fn best_match_against(descriptor: &BriefDescriptor, other_descriptors: &Vec<&BriefDescriptor>) -> usize {
        let (min_idx,_) 
            = other_descriptors.iter().enumerate()
                               .map(|(idx,x)| (idx,descriptor.bit_vector.hamming_distance(&x.bit_vector)))
                               .fold((std::usize::MAX,std::u64::MAX),|(min_idx,min_value),(idx,value)| -> (usize,u64) {
                                   if value < min_value {
                                       (idx,value)
                                   } else {
                                       (min_idx,min_value)
                                   }
                               });

        min_idx               
    }

    pub fn new(image: &Image, orb_feature: &OrbFeature, n: usize, brief_s: usize) -> Option<(BriefDescriptor,Vec<Point<usize>>,Vec<Point<usize>>)> {
        let mut samples_a = Vec::<Point<usize>>::with_capacity(n);
        let mut samples_b = Vec::<Point<usize>>::with_capacity(n);

        let rotation_matrix = rotation_matrix_2d_from_orientation(orb_feature.orientation);
        let mut samples_delta_a = DMatrix::<Float>::zeros(2,n);
        let mut samples_delta_b = DMatrix::<Float>::zeros(2,n);

        let mut bit_vector = BitVector::new(n);

        let std_dev = (brief_s as Float)/5.0;
        let mut sampling_thread = rand::thread_rng();
        let normal_distribution = Normal::new(0.0,std_dev).unwrap();

        let patch_radius = (brief_s-1)/2;
        let top_left_r = orb_feature.location.y as isize-patch_radius as isize;
        let top_left_c = orb_feature.location.x as isize-patch_radius as isize;
        let bottom_right_r = orb_feature.location.y+patch_radius;
        let bottom_right_c = orb_feature.location.x+patch_radius;

        if top_left_r < 0 || top_left_c < 0 || bottom_right_r >= image.buffer.nrows() || bottom_right_c >= image.buffer.ncols() {
            None
        } else {

            //TODO: this is wrong. Precompute sample pairs and rotate them by 2Pi/30 increments making a histogram
            for i in 0..n {
                let (delta_a,delta_b) = BriefDescriptor::generate_sample_pair(&mut sampling_thread,&normal_distribution);
                samples_delta_a[(0,i)] = delta_a.x;
                samples_delta_a[(1,i)] = delta_a.y;
                samples_delta_b[(0,i)] = delta_b.x;
                samples_delta_b[(1,i)] = delta_b.y;

            }   

            let samples_a_rotated = rotation_matrix*samples_delta_a;
            let samples_b_rotated = rotation_matrix*samples_delta_b;

            for i in 0..n{
                // Rotation may make samples go out of bounds
                let a_float = Point::<Float>{x: orb_feature.location.x as Float + samples_a_rotated[(0,i)],y: orb_feature.location.y as Float + samples_a_rotated[(1,i)] };
                let b_float = Point::<Float>{x: orb_feature.location.x as Float + samples_b_rotated[(0,i)],y: orb_feature.location.y as Float + samples_b_rotated[(1,i)] };

                let a = BriefDescriptor::clamp_to_image(&image.buffer,&a_float);
                let b = BriefDescriptor::clamp_to_image(&image.buffer,&b_float);

                bit_vector.add_value(BriefDescriptor::bit_value(&image.buffer,&a, &b));
                samples_a.push(a);
                samples_b.push(b);
            }
    
            Some((BriefDescriptor{bit_vector},samples_a,samples_b))
        }



    }

    fn generate_sample_pair(sampling_thread: &mut ThreadRng,normal_dist: &Normal<Float>) -> (Point<Float>,Point<Float>) {


        let a_x = normal_dist.sample(sampling_thread);
        let a_y = normal_dist.sample(sampling_thread);

        let b_x = normal_dist.sample(sampling_thread);
        let b_y = normal_dist.sample(sampling_thread);


        (Point{x: a_x, y: a_y},Point{x: b_x, y: b_y})

    }

    fn bit_value(image_buffer: &DMatrix<Float>, a: &Point<usize>, b: &Point<usize>) -> u64 {
        let intensity_a = image_buffer[(a.y,a.x)];
        let intensity_b = image_buffer[(b.y,b.x)];

        match intensity_a < intensity_b {
            true => 1,
            _ => 0
        }
    }

    fn clamp_to_image(image_buffer: &DMatrix<Float>, p: &Point<Float>) -> Point<usize> {
        let y = match p.y.trunc() {
            r if r < 0.0 => 0,
            r if r as usize >= image_buffer.nrows() => image_buffer.nrows()-1,
            r => r as usize
        };
        let x = match p.x.trunc() {
            c if c < 0.0 => 0,
            c if c as usize >= image_buffer.ncols() => image_buffer.ncols()-1,
            c => c as usize
        };

        Point::<usize> {x,y}
        
    }
}



