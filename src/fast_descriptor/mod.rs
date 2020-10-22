use crate::features::geometry::{circle::Circle,offset::Offset};
use crate::image::Image;
use crate::Float;

#[derive(Debug,Clone)]
pub struct FastDescriptor {
    pub x_center: usize,
    pub y_center: usize,
    pub starting_offsets: [Offset;4],
    pub continuous_offsets: Vec<Offset>

}

impl FastDescriptor {
    pub fn new(circle: &Circle) -> FastDescriptor {
        let starting_offsets = [circle.offsets[0],circle.offsets[1],circle.offsets[2],circle.offsets[3]];
        let mut positive_y_offset = Vec::<Offset>::with_capacity(circle.offsets.len()/2);
        let mut negative_y_offset = Vec::<Offset>::with_capacity(circle.offsets.len()/2);
        
        for offset in &circle.offsets {
            match offset {
                val if val.y > 0 || val.y == 0 && val.x > 0 => positive_y_offset.push(*val),
                val => negative_y_offset.push(*val)
            };

            positive_y_offset.sort_unstable_by(|a,b| b.cmp(a));
            negative_y_offset.sort_unstable_by(|a,b| a.cmp(b));

        }

        FastDescriptor {x_center:circle.x_center, y_center: circle.y_center,starting_offsets,continuous_offsets:[positive_y_offset,negative_y_offset].concat()}
    }

    pub fn accept(image: &Image, descriptor: &FastDescriptor,threshold_factor: Float) -> bool {

        let sample_intensity = image.buffer[(descriptor.y_center,descriptor.x_center)];
        let t = sample_intensity*threshold_factor;
        let cutoff_max = sample_intensity + t;
        let cutoff_min = sample_intensity - t;

        let number_of_accepted_starting_offsets: usize
            = descriptor.starting_offsets.iter()
            .map(|x| FastDescriptor::sample(image, descriptor.y_center as isize, descriptor.x_center as isize, x))
            .map(|x| !FastDescriptor::within_range(x, cutoff_min, cutoff_max))
            .map(|x| x as usize)
            .sum();

        match number_of_accepted_starting_offsets {
            n if n >= 3 => {
                let mut segment_flag = false;
                for i in 0..descriptor.continuous_offsets.len() {
                    let slice = descriptor.get_wrapping_slice(i, 12);
                    segment_flag = slice.iter()
                    .map(|x| FastDescriptor::sample(image, descriptor.y_center as isize, descriptor.x_center as isize, x))
                    .map(|x| !FastDescriptor::within_range(x, cutoff_min, cutoff_max))
                    .all(|x| x);
                    if segment_flag {
                        break;
                    }
                };
                segment_flag
            },
            _ => false
        }

    }

    fn sample(image: &Image, y_center: isize, x_center: isize, offset: &Offset) -> Float {
        image.buffer[((y_center + offset.y) as usize, (x_center + offset.x) as usize)]
    }

    fn within_range(sample: Float,cutoff_min: Float, cutoff_max: Float) -> bool {
        sample >= cutoff_min && sample <= cutoff_max
    }

    fn get_wrapping_slice(&self, start: usize, size: usize) -> Vec<Offset> {
        let len = self.continuous_offsets.len();
        assert!(size <= len);

        let total = start + size;
        if total < len {
            self.continuous_offsets[start..total].to_vec()
        } else {
            let left_over = len - total;
            [self.continuous_offsets[start..len].to_vec(),self.continuous_offsets[0..left_over].to_vec()].concat()
        }

    }

    pub fn print_continuous_offsets(descriptor: &FastDescriptor) -> () {
        for offset in &descriptor.continuous_offsets {
            println!("{:?}",offset);
        } 
    }


}