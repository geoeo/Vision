use crate::features::geometry::{circle::{Circle,circle_bresenham},offset::Offset};
use crate::image::Image;
use crate::{float,Float};

#[derive(Debug,Clone)]
pub struct FastDescriptor {
    pub x_center: usize,
    pub y_center: usize,
    pub radius: usize,
    pub starting_offsets: [Offset;4],
    pub continuous_offsets: Vec<Offset>

}

impl FastDescriptor {

    pub fn new(x_center: usize, y_center: usize, radius: usize) -> FastDescriptor {
        FastDescriptor::from_circle(&circle_bresenham(x_center,y_center,radius))
    }

    pub fn from_circle(circle: &Circle) -> FastDescriptor {
        let circle_geometry = &circle.geometry;
        let starting_offsets = [circle_geometry.offsets[0],circle_geometry.offsets[1],circle_geometry.offsets[2],circle_geometry.offsets[3]];
        let mut positive_y_offset = Vec::<Offset>::with_capacity(circle_geometry.offsets.len()/2);
        let mut negative_y_offset = Vec::<Offset>::with_capacity(circle_geometry.offsets.len()/2);
        
        for offset in &circle_geometry.offsets {
            match offset {
                val if val.y > 0 || val.y == 0 && val.x > 0 => positive_y_offset.push(*val),
                val => negative_y_offset.push(*val)
            };

            positive_y_offset.sort_unstable_by(|a,b| b.cmp(a));
            negative_y_offset.sort_unstable_by(|a,b| a.cmp(b));

        }

        let mut continuous_offsets = [positive_y_offset,negative_y_offset].concat();
        continuous_offsets.dedup();

        FastDescriptor {x_center:circle_geometry.x_center, y_center: circle_geometry.y_center,radius: circle.radius, starting_offsets,continuous_offsets}
    }

    pub fn accept(image: &Image, descriptor: &FastDescriptor, threshold_factor: Float, n: usize) -> (Option<usize>,Float) {

        if (descriptor.x_center as isize - descriptor.radius as isize) < 0 || descriptor.x_center + descriptor.radius >= image.buffer.ncols() ||
           (descriptor.y_center as isize - descriptor.radius as isize) < 0 || descriptor.y_center + descriptor.radius >= image.buffer.nrows() {
               return (None,float::MIN);
           }

        let sample_intensity = image.buffer[(descriptor.y_center,descriptor.x_center)];
        let t = sample_intensity*threshold_factor;
        let cutoff_max = sample_intensity + t;
        let cutoff_min = sample_intensity - t;

        let number_of_accepted_starting_offsets: usize
            = descriptor.starting_offsets.iter()
            .map(|x| FastDescriptor::sample(image, descriptor.y_center as isize, descriptor.x_center as isize, x))
            .map(|x| FastDescriptor::outside_range(x, cutoff_min, cutoff_max))
            .map(|x| x as usize)
            .sum();

        let perimiter_samples: Vec<Float> = descriptor.continuous_offsets.iter().map(|x| FastDescriptor::sample(image, descriptor.y_center as isize, descriptor.x_center as isize, x)).collect();
        let flagged_scores: Vec<(bool,Float)> = perimiter_samples.iter().map(|&x| (FastDescriptor::outside_range(x, cutoff_min, cutoff_max),FastDescriptor::score(sample_intensity, x, t))).collect();

        match number_of_accepted_starting_offsets {
            count if count >= 3 => {

                let mut max_score = float::MIN;
                let mut max_score_index: Option<usize> = None;
                for i in 0..descriptor.continuous_offsets.len() {
                    let index_slice = descriptor.get_wrapping_index(i, n);
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

    pub fn compute_valid_descriptors(image: &Image, radius: usize,  threshold_factor: Float, n: usize, x_grid: usize, y_grid: usize) -> Vec<(FastDescriptor,usize)> {

        let mut result = Vec::<(FastDescriptor,usize)>::new();
        // for r in (0..image.buffer.nrows()).step_by(1) {
        //     for c in (0..image.buffer.ncols()).step_by(1) {    
        //         let descriptor = FastDescriptor::new(c, r, radius);
        //         let (start_option,score) = FastDescriptor::accept(image, &descriptor, threshold_factor, n);

        //         if start_option.is_some() && score > grid_max_score {
        //             grid_points.push((descriptor,start_option.unwrap()));
        //             result.push((descriptor,start_option.unwrap()));
        //         }

        //         // Will be wrong -  fix predicate % iteration
        //         if c%x_grid == 0 && r&y_grid == 0 && c != 0 && r != 0 {


        //             result.extend(grid_points.clone());
        //             grid_points.clear();
        //             let mut grid_max_score = float::MIN;
        //         }
    
        //     }
        // }

        for r in (0..image.buffer.nrows()).step_by(y_grid) {
            for c in (0..image.buffer.ncols()).step_by(x_grid) {

                let mut grid_max_option: Option<(FastDescriptor,usize)> = None;
                let mut grid_max_score = float::MIN;
                
                for r_grid in r..r+y_grid {
                    for c_grid in c..c+x_grid {
                        let descriptor = FastDescriptor::new(c_grid, r_grid, radius);
                        let (start_option,score) = FastDescriptor::accept(image, &descriptor, threshold_factor, n);
                        if start_option.is_some() && score > grid_max_score {
                            grid_max_score = score;
                            grid_max_option = Some((descriptor,start_option.unwrap()));
                        }

                    }
                }

                if grid_max_option.is_some() {
                    result.push(grid_max_option.unwrap());
                }


    
            }
        }

        result

    }

    fn sample(image: &Image, y_center: isize, x_center: isize, offset: &Offset) -> Float {
        image.buffer[((y_center + offset.y) as usize, (x_center + offset.x) as usize)]
    }

    fn outside_range(perimiter_sample: Float, cutoff_min: Float, cutoff_max: Float) -> bool {
        FastDescriptor::is_bright(perimiter_sample, cutoff_max) || FastDescriptor::is_dark(perimiter_sample, cutoff_min)
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

    pub fn print_continuous_offsets(descriptor: &FastDescriptor) -> () {
        for offset in &descriptor.continuous_offsets {
            println!("{:?}",offset);
        } 
    }


}