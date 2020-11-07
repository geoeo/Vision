extern crate rand_distr;
extern crate rand;
extern crate nalgebra as na;

use rand::prelude::ThreadRng;
use rand_distr::{Normal,Distribution};
use na::{DMatrixSlice,DMatrix};
use crate::image::Image;
use crate::features::{geometry::point::Point, orb_feature::OrbFeature};
use crate::Float;
use crate::numerics::rotation_matrix_2d_from_orientation;
use self::bit_vector::BitVector;


pub mod bit_vector;

#[derive(Debug,Clone)]
pub struct BriefDescriptor {
    samples_a: Vec<Point<usize>>,
    samples_b: Vec<Point<usize>>,
    bit_vector: BitVector

}

impl BriefDescriptor {

    pub fn new(image: &Image, orb_feature: &OrbFeature, n: usize, brief_s: usize) -> Option<BriefDescriptor> {
        let mut samples_a_vec = Vec::<Point<usize>>::with_capacity(n);
        let mut samples_b_vec = Vec::<Point<usize>>::with_capacity(n);

        let rotation_matrix = rotation_matrix_2d_from_orientation(orb_feature.orientation);
        let mut samples_a = DMatrix::<Float>::zeros(2,n);
        let mut samples_b = DMatrix::<Float>::zeros(2,n);

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

            for i in 0..n {
                let (a,b) = BriefDescriptor::generate_sample_pair(&orb_feature.location, &mut sampling_thread,&normal_distribution);
                samples_a[(0,i)] = a.x;
                samples_a[(1,i)] = a.y;
                samples_b[(0,i)] = b.x;
                samples_b[(1,i)] = b.y;

            }   

            let samples_a_rotated = rotation_matrix*samples_a;
            let samples_b_rotated = rotation_matrix*samples_b;
            let slice = image.buffer.slice((top_left_r as usize,top_left_c as usize),(bottom_right_r,bottom_right_c));

            for i in 0..n{
                let a = Point::<usize>{x: samples_a_rotated[(0,i)].trunc() as usize,y: samples_a_rotated[(1,i)].trunc() as usize };
                let b = Point::<usize>{x: samples_b_rotated[(0,i)].trunc() as usize,y: samples_b_rotated[(1,i)].trunc() as usize };
                bit_vector.add_value(BriefDescriptor::bit_value(&slice,&a, &b));
                samples_a_vec.push(a);
                samples_b_vec.push(b);
            }
    
            Some(BriefDescriptor{samples_a: samples_a_vec,samples_b: samples_b_vec,bit_vector})
        }



    }

    fn generate_sample_pair(center: &Point<usize>, sampling_thread: &mut ThreadRng,normal_dist: &Normal<Float>) -> (Point<Float>,Point<Float>) {


        let a_x = center.x as Float + normal_dist.sample(sampling_thread);
        let a_y = center.y as Float + normal_dist.sample(sampling_thread);

        let b_x = center.x as Float + normal_dist.sample(sampling_thread);
        let b_y = center.y as Float + normal_dist.sample(sampling_thread);


        (Point{x: a_x, y: a_y},Point{x: b_x, y: b_y})

    }

    fn bit_value(image_buffer: &DMatrixSlice<Float>, a: &Point<usize>, b: &Point<usize>) -> u64 {
        let intensity_a = image_buffer[(a.y,a.x)];
        let intensity_b = image_buffer[(b.y,b.x)];

        match intensity_a < intensity_b {
            true => 1,
            _ => 0
        }
    }
}



