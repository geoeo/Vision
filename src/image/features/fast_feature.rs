use crate::image::features::{Feature, geometry::{point::Point,shape::circle::{Circle,circle_bresenham},Offset}};
use crate::image::pyramid::orb::orb_runtime_parameters::OrbRuntimeParameters;
use crate::image::Image;
use crate::{float,Float};

#[derive(Debug,Clone)]
pub struct FastFeature {
    pub location: Point<usize>,
    pub radius: usize,
    pub starting_offsets: [Offset;4],
    pub continuous_offsets: Vec<Offset>,
    pub id: Option<u64>

}

impl Feature for FastFeature {

    fn get_x_image_float(&self) -> Float { self.get_x_image() as Float}
    fn get_y_image_float(&self) -> Float { self.get_y_image() as Float}

    fn get_x_image(&self) -> usize {
        self.location.x
    }

    fn get_y_image(&self) -> usize {
        self.location.y
    }

    fn get_closest_sigma_level(&self) -> usize {
        0
    }

    fn apply_normalisation(&self, _: &nalgebra::Matrix3<Float>, _: Float) -> Self {
        panic!("TODO: FastFeature apply_normalisation")
    }

}

impl FastFeature {

    fn new(x_center: usize, y_center: usize, radius: usize) -> FastFeature {
        FastFeature::from_circle(&circle_bresenham(x_center,y_center,radius))
    }

    fn from_circle(circle: &Circle) -> FastFeature {
        let circle_geometry = &circle.shape;
        let starting_offsets = [circle_geometry.offsets[0],circle_geometry.offsets[1],circle_geometry.offsets[2],circle_geometry.offsets[3]];
        let mut positive_y_offset = Vec::<Offset>::with_capacity(circle_geometry.offsets.len()/2);
        let mut negative_y_offset = Vec::<Offset>::with_capacity(circle_geometry.offsets.len()/2);
        let mut pos_zero = Vec::<Offset>::with_capacity(1);
        let mut neg_zero = Vec::<Offset>::with_capacity(1);
        
        for offset in &circle_geometry.offsets {
            match offset {
                val if val.y > 0 => positive_y_offset.push(*val),
                val if val.y < 0 => negative_y_offset.push(*val),
                val if val.y == 0 && val.x > 0 => pos_zero.push(*val),
                val => neg_zero.push(*val)
            };

            positive_y_offset.sort_unstable_by(|a,b| b.cmp(a));
            negative_y_offset.sort_unstable_by(|a,b| a.cmp(b));

        }



        let mut continuous_offsets = [pos_zero,positive_y_offset,neg_zero,negative_y_offset].concat();
        continuous_offsets.dedup();

        FastFeature {location: circle_geometry.center,radius: circle.radius, starting_offsets,continuous_offsets, id: None }
    }

    fn accept(image: &Image, feature: &FastFeature, threshold_factor: Float, consecutive_pixels: usize) -> (Option<usize>,Float) {

        if (feature.location.x as isize - feature.radius as isize) < 0 || feature.location.x + feature.radius >= image.buffer.ncols() ||
           (feature.location.y as isize - feature.radius as isize) < 0 || feature.location.y + feature.radius >= image.buffer.nrows() {
               return (None,float::MIN);
           }

        let sample_intensity = image.buffer[(feature.location.y,feature.location.x)];
        let t = sample_intensity*threshold_factor;
        let cutoff_max = sample_intensity + t;
        let cutoff_min = sample_intensity - t;

        let number_of_accepted_starting_offsets: usize
            = feature.starting_offsets.iter()
            .map(|x| FastFeature::sample(image, feature.location.y as isize, feature.location.x as isize, x))
            .map(|x| FastFeature::outside_range(x, cutoff_min, cutoff_max))
            .map(|x| x as usize)
            .sum();

        let perimiter_samples: Vec<Float> = feature.continuous_offsets.iter().map(|x| FastFeature::sample(image, feature.location.y as isize, feature.location.x as isize, x)).collect();
        let flagged_scores: Vec<(bool,Float)> = perimiter_samples.iter().map(|&x| (FastFeature::outside_range(x, cutoff_min, cutoff_max),FastFeature::score(sample_intensity, x, t))).collect();

        match number_of_accepted_starting_offsets {
            count if count >= 3 => {

                let mut max_score = float::MIN;
                let mut max_score_index: Option<usize> = None;
                for i in 0..feature.continuous_offsets.len() {
                    let index_slice = feature.get_wrapping_index(i, consecutive_pixels);
                    let (all_passing,total_score) = index_slice.iter()
                    .map(|&x| flagged_scores[x])
                    .fold((true,0.0),|(b_acc,s_acc),x| (b_acc && x.0,s_acc + x.1));

                    if all_passing {
                        if total_score > max_score {
                            max_score = total_score;
                            max_score_index = Some(i);
                        }
                    }
                };

                (max_score_index,max_score)
            },
            _ => (None,float::MIN)
        }

    }

    //TODO: performance offender
    pub fn compute_valid_feature(image: &Image, radius: usize,  threshold_factor: Float, features_per_grid: usize, consecutive_pixels: usize, x_grid_start: usize, y_grid_start: usize, x_grid_size: usize, y_grid_size: usize) -> Vec<FastFeature> /*Option<(FastFeature,usize)>*/ {
        let mut features = Vec::<(FastFeature,Float)>::with_capacity(y_grid_size*x_grid_size);
        
        for r_grid in y_grid_start..y_grid_start+y_grid_size {
            for c_grid in x_grid_start..x_grid_start+x_grid_size {
                let feature = FastFeature::new(c_grid, r_grid, radius);
                let (start_option,score) = FastFeature::accept(image, &feature, threshold_factor, consecutive_pixels);

                if start_option.is_some() {
                    features.push((feature,score));
                }

            }
        }

        features.sort_unstable_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        let n = std::cmp::min(features_per_grid,features.len());
        features.into_iter().take(n).map(|x| x.0).collect::<Vec<FastFeature>>()

    }


    pub fn compute_valid_features(image: &Image,octave_idx: i32, runtime_parameters: &OrbRuntimeParameters) -> Vec<FastFeature> {


        let orig_offset = runtime_parameters.fast_offsets;
        let offset_scale = runtime_parameters.fast_offset_scale_base.powi(octave_idx) as Float;
        let x_offset_scaled = (orig_offset.0 as Float / offset_scale).trunc() as usize;
        let y_offset_scaled = (orig_offset.1 as Float / offset_scale).trunc() as usize;
        
        let octave_scale = runtime_parameters.fast_grid_size_scale_base.powi(octave_idx);
        let scale_grid_size = ((runtime_parameters.fast_grid_size.0 as Float * octave_scale).trunc() as usize, (runtime_parameters.fast_grid_size.1 as Float * octave_scale).trunc() as usize);

        let x_grid = scale_grid_size.0;
        let y_grid = scale_grid_size.1;
        let x_offset = x_offset_scaled;
        let y_offset = y_offset_scaled;
        
        //TODO: initialize capcity properly
        let mut result = Vec::<FastFeature>::new();
        for r in (y_offset..image.buffer.nrows() - y_offset).step_by(y_grid) {
            for c in (x_offset..image.buffer.ncols() - x_offset).step_by(x_grid) {

                match FastFeature::compute_valid_feature(image,runtime_parameters.fast_circle_radius,runtime_parameters.fast_threshold_factor, runtime_parameters.fast_features_per_grid ,runtime_parameters.fast_consecutive_pixels,c,r,x_grid,y_grid) {
                    mut vec if !vec.is_empty()=> result.append(&mut vec),
                    _ => ()
                }

            }
        }

        result

    }

    fn sample(image: &Image, y_center: isize, x_center: isize, offset: &Offset) -> Float {
        image.buffer[((y_center + offset.y) as usize, (x_center + offset.x) as usize)]
    }

    fn outside_range(perimiter_sample: Float, cutoff_min: Float, cutoff_max: Float) -> bool {
        FastFeature::is_bright(perimiter_sample, cutoff_max) || FastFeature::is_dark(perimiter_sample, cutoff_min)
    }

    fn score(sample :Float, perimiter_sample: Float, t: Float) -> Float {
        (sample-perimiter_sample).abs() - t
    }

    fn is_bright(perimiter_sample: Float, cutoff_max: Float) -> bool {
        perimiter_sample >= cutoff_max
    }

    fn is_dark(perimiter_sample: Float, cutoff_min: Float) -> bool {
        perimiter_sample <= cutoff_min
    }

    pub fn get_wrapping_slice(&self, start: usize, size: usize) -> Vec<Offset> {
        let len = self.continuous_offsets.len();
        assert!(size <= len);

        let total = start + size;
        if total < len {
            self.continuous_offsets[start..total].to_vec()
        } else {
            let left_over = total - len;
            [self.continuous_offsets[start..len].to_vec(),self.continuous_offsets[0..left_over].to_vec()].concat()
        }

    }

    pub fn get_wrapping_index(&self, start: usize, size: usize) -> Vec<usize> {
        let len = self.continuous_offsets.len();
        assert!(size <= len);

        let total = start + size;
        if total < len {
            (start..total).collect()
        } else {
            let left_over = total - len;
            [(start..len).collect::<Vec<usize>>(),(0..left_over).collect()].concat()
        }

    }

    pub fn get_full_circle(&self) -> Circle {
        let continuous_offset_half = self.continuous_offsets.len()/2;

        let mut offsets = Vec::<Offset>::new();
        let mut pos_offsets = Vec::<Offset>::new();
        let mut neg_offsets = Vec::<Offset>::new();

        offsets.push(self.continuous_offsets[0]);
        offsets.push(self.continuous_offsets[continuous_offset_half]);

        for i in 1..continuous_offset_half {
            let pos_y_idx = i;
            let neg_y_idy = continuous_offset_half + i;

            let pos_y_offset = self.continuous_offsets[pos_y_idx];
            let neg_y_offset = self.continuous_offsets[neg_y_idy];

            pos_offsets.push(pos_y_offset);
            neg_offsets.push(neg_y_offset);
        }
        neg_offsets.reverse();
        let offset_pairs_iter = pos_offsets.iter().zip(neg_offsets.iter());

        for (p,n) in offset_pairs_iter {
            for y in n.y..p.y+1{
                offsets.push(Offset{x:p.x,y});
            }

        }

        Circle::new(self.location.x,self.location.y, self.radius, offsets)
    }

    pub fn get_all_points_in_radius(&self) -> (Vec<Point<usize>>, Point<usize>) {

        let mut points = Vec::<Point<usize>>::with_capacity(self.radius.pow(2));
        let radius_singed = self.radius as isize;
        let location_x_signed = self.location.x as isize;
        let location_y_signed = self.location.y as isize;

        for i in -radius_singed..radius_singed {
            for j in -radius_singed..radius_singed {
                let new_location_x = location_x_signed+i;
                let new_location_y = location_y_signed+j;
                if new_location_x >= 0 && new_location_y >= 0{
                    points.push(Point::<usize>::new(new_location_x as usize, new_location_y as usize));
                }
            }
        }

        (points, self.location)
    }

    pub fn print_continuous_offsets(feature: &FastFeature) -> () {
        for offset in &feature.continuous_offsets {
            println!("{:?}",offset);
        } 
    }


}