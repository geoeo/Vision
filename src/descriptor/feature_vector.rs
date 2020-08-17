
use crate::descriptor::{ORIENTATION_BINS,DESCRIPTOR_BINS,local_image_descriptor::LocalImageDescriptor};

#[derive(Debug,Clone)]
pub struct FeatureVector {
    data: [[u8;ORIENTATION_BINS];DESCRIPTOR_BINS]
}

impl FeatureVector {
    pub fn new(descriptor: &LocalImageDescriptor) -> FeatureVector {
            //TODO: normalize
            //TODO: Saturate
            //TODO: Renormalize and cast to 8bit ints
            let orientation = [0;ORIENTATION_BINS];
            let data = [orientation;DESCRIPTOR_BINS];
            FeatureVector{data}
        
    }
}