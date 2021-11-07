extern crate nalgebra as na;


use na::{Matrix4,DMatrix};
use crate::Float;
use crate::image::features::{ Feature,Match,orb_feature::OrbFeature};


pub fn eight_point<T: Feature>(matches: &Vec<Match<T>>) -> Matrix4<Float> {


    let number_of_matches = matches.len();
    let A = DMatrix::<Float>::zeros(number_of_matches,9);
    for m in matches {
        // fill A
    }
    panic!("TODO");
}