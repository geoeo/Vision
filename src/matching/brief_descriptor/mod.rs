extern crate rand_distr;
extern crate rand;
extern crate nalgebra as na;

use rand::prelude::ThreadRng;
use rand_distr::{Normal,Distribution};
use na::DMatrix;

use crate::image::Image;
use crate::pyramid::Pyramid;
use crate::features::{geometry::point::Point, orb_feature::OrbFeature};
use crate::{Float,float};
use crate::numerics::rotation_matrix_2d_from_orientation;
use self::bit_vector::BitVector;


pub mod bit_vector;

#[derive(Debug,Clone)]
pub struct BriefDescriptor {
    bit_vector: BitVector
}

impl BriefDescriptor {

    pub fn generate_sample_lookup_tables(n: usize, brief_s: usize) -> Vec<Vec<(Point<Float>,Point<Float>)>> {
        let std_dev = (brief_s as Float)/5.0;
        let mut sampling_thread = rand::thread_rng();
        let normal_distribution = Normal::new(0.0,std_dev).unwrap();

        let mut samples_delta_a = DMatrix::<Float>::zeros(2,n);
        let mut samples_delta_b = DMatrix::<Float>::zeros(2,n);
        let step = 15.0; //TODO: make this a parameter

        let mut lookup_tables = Vec::<Vec<(Point<Float>,Point<Float>)>>::with_capacity(step as usize);

        let table_inc = 2.0*float::consts::PI/step;

        for _ in 0..lookup_tables.capacity() {
            lookup_tables.push(Vec::<(Point<Float>,Point<Float>)>::with_capacity(n));
        }

        for i in 0..n {
            let (delta_a,delta_b) = BriefDescriptor::generate_sample_pair(&mut sampling_thread,&normal_distribution);
            samples_delta_a[(0,i)] = delta_a.x;
            samples_delta_a[(1,i)] = delta_a.y;
            samples_delta_b[(0,i)] = delta_b.x;
            samples_delta_b[(1,i)] = delta_b.y;
        }

        for j in 0..lookup_tables.len(){
            let angle = table_inc*j as Float;
            // We transpose here because the y-axis of a matrix is inverted from the first quadrant of a cartesian plane
            let rotation_matrix = rotation_matrix_2d_from_orientation(angle).transpose();

            let rotated_delta_a = rotation_matrix*&samples_delta_a;
            let rotated_delta_b = rotation_matrix*&samples_delta_b;

            for i in 0..n {
                lookup_tables[j].push((Point::new(rotated_delta_a[(0,i)], rotated_delta_a[(1,i)]),Point::new(rotated_delta_b[(0,i)], rotated_delta_b[(1,i)])));
            }
        }


        lookup_tables

    }

    pub fn generate_sample_lookup_table_pyramid(n: usize, brief_s: usize, octave_count: usize) -> Pyramid<Vec<Vec<(Point<Float>,Point<Float>)>>> {

        let mut octaves: Vec<Vec<Vec<(Point<Float>,Point<Float>)>>> = Vec::with_capacity(octave_count);
        for i in 0..octave_count {
            let scale_base: Float = 2.0;
            let octave_scale = scale_base.powi(i as i32);
            let brief_s_scaled = (brief_s as Float/ octave_scale).round() as usize;

            octaves.push(BriefDescriptor::generate_sample_lookup_tables(n, brief_s_scaled));
            //octaves.push(BriefDescriptor::generate_sample_lookup_tables(n, brief_s));
        }

        Pyramid {octaves}

    }

    pub fn match_descriptors(descriptors_a: &Vec<BriefDescriptor>, descriptors_b: &Vec<BriefDescriptor>, matching_min_threshold: u64) -> Vec<Option<usize>> {
        descriptors_a.iter().map(|x| BriefDescriptor::best_match_against(x, descriptors_b,matching_min_threshold)).collect::<Vec<Option<usize>>>()
    }

    pub fn best_match_against(descriptor: &BriefDescriptor, other_descriptors: &Vec<BriefDescriptor>, matching_min_threshold: u64) -> Option<usize> {
        let (min_idx,_) 
            = other_descriptors.iter().enumerate()
                               .map(|(idx,x)| (idx,descriptor.bit_vector.hamming_distance(&x.bit_vector)))
                               .fold((std::usize::MAX,std::u64::MAX),|(min_idx,min_value),(idx,value)| -> (usize,u64) {
                                   if value < min_value && value < matching_min_threshold { 
                                       (idx,value)
                                   } else {
                                       (min_idx,min_value)
                                   }
                               });

        match min_idx  {
            std::usize::MAX => None,
            v => Some(v)
        }             
    }

    pub fn new(image: &Image, orb_feature: &OrbFeature, n: usize, brief_s: usize, octave_idx: usize, sample_lookup_tables: &Vec<Vec<(Point<Float>,Point<Float>)>>) -> Option<BriefDescriptor> {
        let mut samples_a = Vec::<Point<usize>>::with_capacity(n);
        let mut samples_b = Vec::<Point<usize>>::with_capacity(n);
        let mut bit_vector = BitVector::new(n);

        let scale_base: Float = 2.0;
        let octave_scale = scale_base.powi(octave_idx as i32);
        let brief_s_scaled = (brief_s as Float/ octave_scale).round() as usize;


        //let patch_radius = (brief_s-1)/2;
        let patch_radius = (brief_s_scaled-1)/2;
        let top_left_r = orb_feature.location.y as isize-patch_radius as isize;
        let top_left_c = orb_feature.location.x as isize-patch_radius as isize;
        let bottom_right_r = orb_feature.location.y+patch_radius;
        let bottom_right_c = orb_feature.location.x+patch_radius;

        if top_left_r < 0 || top_left_c < 0 || bottom_right_r >= image.buffer.nrows() || bottom_right_c >= image.buffer.ncols() {
            None
        } else {

            let max_bin = (sample_lookup_tables.len()-1) as Float;
            let idx_as_float = orb_feature.orientation / (2.0*float::consts::PI/max_bin);
            let sample_pair_idx =  idx_as_float.round() as usize;
            let samples_pattern = &sample_lookup_tables[sample_pair_idx];
            

            for i in 0..n{

                let (sample_a,sample_b) = samples_pattern[i];
                // Rotation may make samples go out of bounds -> Check this
                let a_float = Point::<Float>{x: orb_feature.location.x as Float + sample_a.x, y: orb_feature.location.y as Float + sample_a.y};
                let b_float = Point::<Float>{x: orb_feature.location.x as Float + sample_b.x, y: orb_feature.location.y as Float + sample_b.y};

                let a = BriefDescriptor::clamp_to_image(&image.buffer,&a_float);
                let b = BriefDescriptor::clamp_to_image(&image.buffer,&b_float);

                bit_vector.add_value(BriefDescriptor::bit_value(&image.buffer,&a, &b));
                samples_a.push(a);
                samples_b.push(b);
            }
    
            Some(BriefDescriptor{bit_vector})
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



