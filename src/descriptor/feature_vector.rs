
use crate::descriptor::{ORIENTATION_BINS,DESCRIPTOR_BINS,local_image_descriptor::LocalImageDescriptor};

#[derive(Debug,Clone)]
pub struct FeatureVector {
    data: [[u8;ORIENTATION_BINS];DESCRIPTOR_BINS]
}

impl FeatureVector {
    pub fn new(descriptor: &LocalImageDescriptor) -> FeatureVector {
            let orientation = [0;ORIENTATION_BINS];
            let mut data = [orientation;DESCRIPTOR_BINS];


            for i in 0..DESCRIPTOR_BINS{
                let descriptor_vector = &descriptor.descriptor_vector[i];
                let magnitude = descriptor_vector.squared_magnitude.sqrt();
                for j in 0..ORIENTATION_BINS {
                    let descriptor_value_norm = descriptor_vector.bins[j]/magnitude;
                    let saturated_value = match (descriptor_value_norm.floor(),0.2*magnitude) {
                        (a,b) if a <= b => a,
                        (_,b) => b
                    };
                    let quantized_value = std::cmp::min(saturated_value.floor() as u8,255);
                    data[i][j] = quantized_value;
                }
            }
            FeatureVector{data}
        
    }
}