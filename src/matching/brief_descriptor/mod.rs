use crate::image::Image;
use crate::features::geometry::point::Point;
use self::bit_vector::BitVector;

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

        for _ in 0..n {

            let (a,b) = BriefDescriptor::generate_sample_pair(image,s);

            bit_vector.add_value(BriefDescriptor::bit_value(image,&a, &b));
            samples_a.push(a);
            samples_b.push(b);


        }

        BriefDescriptor{samples_a,samples_b,bit_vector}

    }

    fn generate_sample_pair(image: &Image, s: usize) -> (Point,Point) {

        panic!("not implemented")

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



