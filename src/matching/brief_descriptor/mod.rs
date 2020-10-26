use crate::features::geometry::point::Point;
use self::bit_vector::BitVector;

pub mod bit_vector;

#[derive(Debug,Clone)]
pub struct BriefDescriptor {
    samples_a: Vec<Point>,
    samples_b: Vec<Point>,
    bit_vector: BitVector

}



