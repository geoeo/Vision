extern crate rand_distr;
extern crate rand;
extern crate nalgebra as na;

use rand::prelude::*;
use rand_distr::{Normal,Distribution};
use na::DMatrix;

use crate::image::Image;
use crate::pyramid::{Pyramid, orb::orb_runtime_parameters::OrbRuntimeParameters};
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

    pub fn generate_sample_lookup_tables(runtime_parameters: &OrbRuntimeParameters, octave_idx: i32) -> Vec<Vec<(Point<Float>,Point<Float>)>> {


        let octave_scale = runtime_parameters.brief_s_scale_base.powi(octave_idx);
        let brief_s_scaled = (runtime_parameters.brief_s as Float/ octave_scale).round();

        let std_dev = (brief_s_scaled)/5.0;


        let mut sampling_thread = rand::rngs::SmallRng::seed_from_u64(0x0DDB1A5ECBAD5EEDu64); 
        let normal_distribution = Normal::new(0.0,std_dev).unwrap();

        let mut samples_delta_a = DMatrix::<Float>::zeros(2,runtime_parameters.brief_n);
        let mut samples_delta_b = DMatrix::<Float>::zeros(2,runtime_parameters.brief_n);
        let step = runtime_parameters.brief_lookup_table_step; 

        let mut lookup_tables = Vec::<Vec<(Point<Float>,Point<Float>)>>::with_capacity(step as usize);

        let table_inc = 2.0*float::consts::PI/step;

        for _ in 0..lookup_tables.capacity() {
            lookup_tables.push(Vec::<(Point<Float>,Point<Float>)>::with_capacity(runtime_parameters.brief_n));
        }

        for i in 0..runtime_parameters.brief_n {
            let (delta_a,delta_b) = BriefDescriptor::generate_sample_pair(&mut sampling_thread,&normal_distribution);
            samples_delta_a[(0,i)] = delta_a.x;
            samples_delta_a[(1,i)] = delta_a.y;
            samples_delta_b[(0,i)] = delta_b.x;
            samples_delta_b[(1,i)] = delta_b.y;
        }

        for j in 0..step as usize{
            let angle = table_inc*j as Float;

            //TODO: check this
            // We transpose here because the y-axis of a matrix is inverted from the first quadrant of a cartesian plane
            //let rotation_matrix = rotation_matrix_2d_from_orientation(angle).transpose();
            let rotation_matrix = rotation_matrix_2d_from_orientation(angle);

            let rotated_delta_a = rotation_matrix*&samples_delta_a;
            let rotated_delta_b = rotation_matrix*&samples_delta_b;

            for i in 0..runtime_parameters.brief_n {
                //TODO: check this
                lookup_tables[j].push((Point::new(rotated_delta_a[(0,i)], rotated_delta_a[(1,i)]),Point::new(rotated_delta_b[(0,i)], rotated_delta_b[(1,i)])));
            }
        }


        lookup_tables

    }

    pub fn generate_sample_lookup_table_pyramid(runtime_parameters: &OrbRuntimeParameters, octave_count: usize) -> Pyramid<Vec<Vec<(Point<Float>,Point<Float>)>>> {

        let mut octaves: Vec<Vec<Vec<(Point<Float>,Point<Float>)>>> = Vec::with_capacity(octave_count);
        for i in 0..octave_count {
            octaves.push(BriefDescriptor::generate_sample_lookup_tables(runtime_parameters, i as i32));
        }

        Pyramid {octaves}

    }

    pub fn sorted_match_descriptors(descriptors_a: &Vec<BriefDescriptor>, descriptors_b: &Vec<BriefDescriptor>, matching_min_threshold: u64) -> Vec<Option<Vec<(usize, u64)>>> {
        descriptors_a.iter().map(|x| BriefDescriptor::sorted_matches_against(x, descriptors_b,matching_min_threshold)).collect::<Vec<Option<Vec<(usize, u64)>>>>()
    }

    pub fn sorted_matches_against(descriptor: &BriefDescriptor, other_descriptors: &Vec<BriefDescriptor>, matching_min_threshold: u64) -> Option<Vec<(usize, u64)>> {
        let indicies_hamming_distances
            = other_descriptors
            .iter()
            .enumerate()
            .map(|(idx,x)| (idx,descriptor.bit_vector.hamming_distance(&x.bit_vector))).collect::<Vec<(usize, u64)>>();

        match indicies_hamming_distances {
            vec if vec.is_empty() => None,
            vec if vec[0].1 > matching_min_threshold => None,
            mut vec => {
                vec.sort_unstable_by(|a,b|  a.1.partial_cmp(&b.1).unwrap());
                Some(vec)
            }
        }
    }

    pub fn new(image: &Image, orb_feature: &OrbFeature, runtime_parameters: &OrbRuntimeParameters, octave_idx: usize, sample_lookup_tables: &Vec<Vec<(Point<Float>,Point<Float>)>>) -> Option<BriefDescriptor> {
        let mut samples_a = Vec::<Point<usize>>::with_capacity(runtime_parameters.brief_n);
        let mut samples_b = Vec::<Point<usize>>::with_capacity(runtime_parameters.brief_n);
        let mut bit_vector = BitVector::new(runtime_parameters.brief_n);

        let octave_scale = runtime_parameters.brief_s_scale_base.powi(octave_idx as i32);
        let brief_s_scaled = (runtime_parameters.brief_s as Float/ octave_scale).round() as usize;


        //let patch_radius = (brief_s-1)/2;
        let patch_radius = (brief_s_scaled-1)/2;
        let top_left_r = orb_feature.location.y as isize - patch_radius as isize;
        let top_left_c = orb_feature.location.x as isize - patch_radius as isize;
        let bottom_right_r = orb_feature.location.y+patch_radius;
        let bottom_right_c = orb_feature.location.x+patch_radius;

        if top_left_r < 0 || top_left_c < 0 || bottom_right_r >= image.buffer.nrows() || bottom_right_c >= image.buffer.ncols() {
            None
        } else {

            let max_bin = (sample_lookup_tables.len()-1) as Float;
            let idx_as_float = orb_feature.orientation / (2.0*float::consts::PI/max_bin);
            let sample_pair_idx =  idx_as_float.round() as usize;

            // let max_bin = (sample_lookup_tables.len()) as Float;
            // let idx_as_float = orb_feature.orientation / (2.0*float::consts::PI/max_bin);
            // let sample_pair_idx =  idx_as_float.trunc() as usize;

            //println!("x {}, y {}, o: {}", orb_feature.location.x, orb_feature.location.y, orb_feature.orientation);
            //println!("{}, {}", orb_feature.location.x, orb_feature.location.y);
            //let sample_pair_idx =  sample_lookup_tables.len()- 1 - idx_as_float.round() as usize;
            let samples_pattern = &sample_lookup_tables[sample_pair_idx];
            

            for (sample_a,sample_b) in samples_pattern{

    
                // Rotation may make samples go out of bounds -> Check this
                let a_float = Point::<Float>{x: orb_feature.location.x as Float + sample_a.x, y: orb_feature.location.y as Float - sample_a.y};
                let b_float = Point::<Float>{x: orb_feature.location.x as Float + sample_b.x, y: orb_feature.location.y as Float - sample_b.y};

                let a = BriefDescriptor::clamp_to_image(&image.buffer,&a_float);
                let b = BriefDescriptor::clamp_to_image(&image.buffer,&b_float);


                bit_vector.add_value(BriefDescriptor::bit_value(&image.buffer,&a, &b));
                samples_a.push(a);
                samples_b.push(b);
            }
    
            Some(BriefDescriptor{bit_vector})
        }



    }

    fn generate_sample_pair(sampling_thread: &mut dyn RngCore,normal_dist: &Normal<Float>) -> (Point<Float>,Point<Float>) {
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



