extern crate rand_distr;
extern crate rand;

use rand::prelude::ThreadRng;
use rand_distr::{Normal,Distribution};
use crate::image::Image;
use crate::features::geometry::point::Point;
use self::bit_vector::BitVector;
use crate::Float;

pub mod bit_vector;

#[derive(Debug,Clone)]
pub struct BriefDescriptor {
    samples_a: Vec<Point>,
    samples_b: Vec<Point>,
    bit_vector: BitVector

}

impl BriefDescriptor {
    pub fn new(image: &Image, n: usize, s: usize) -> BriefDescriptor {

        let mut samples_a = Vec::<Point>::with_capacity(n);
        let mut samples_b = Vec::<Point>::with_capacity(n);
        let mut bit_vector = BitVector::new(n);

        let std_dev = (s as Float)/5.0;
        let mut sampling_thread = rand::thread_rng();
        let normal_distribution = Normal::new(0.0,std_dev).unwrap();


        for _ in 0..n {

            let (a,b) = BriefDescriptor::generate_sample_pair(image, &mut sampling_thread,&normal_distribution);

            bit_vector.add_value(BriefDescriptor::bit_value(image,&a, &b));
            samples_a.push(a);
            samples_b.push(b);


        }

        BriefDescriptor{samples_a,samples_b,bit_vector}

    }

    fn generate_sample_pair(image: &Image, sampling_thread: &mut ThreadRng,normal_dist: &Normal<Float>) -> (Point,Point) {

        let a_x = normal_dist.sample(sampling_thread);
        let a_y = normal_dist.sample(sampling_thread);

        let b_x = normal_dist.sample(sampling_thread);
        let b_y = normal_dist.sample(sampling_thread);

        let a_x_image = match a_x {
            v if v < 0.0 => 0,
            v if v.trunc() as usize >= image.buffer.ncols() => image.buffer.ncols()-1,
            v => v.trunc() as usize
        };

        let a_y_image = match a_y {
            v if v < 0.0 => 0,
            v if v.trunc() as usize >= image.buffer.nrows() => image.buffer.nrows()-1,
            v => v.trunc() as usize
        };

        let b_x_image = match b_x {
            v if v < 0.0 => 0,
            v if v.trunc() as usize >= image.buffer.ncols() => image.buffer.ncols()-1,
            v => v.trunc() as usize
        };

        let b_y_image = match b_y {
            v if v < 0.0 => 0,
            v if v.trunc() as usize >= image.buffer.nrows() => image.buffer.nrows()-1,
            v => v.trunc() as usize
        };


        (Point{x: a_x_image, y: a_y_image},Point{x: b_x_image, y: b_y_image})

    }

    fn bit_value(image: &Image, a: &Point, b: &Point) -> u64 {
        let intensity_a = image.buffer[(a.y,a.x)];
        let intensity_b = image.buffer[(b.y,b.x)];

        match intensity_a < intensity_b {
            true => 1,
            _ => 0
        }
    }
}



