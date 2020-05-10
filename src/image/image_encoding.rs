use crate::Float;

#[repr(u8)]
#[derive(Debug,Copy,Clone,PartialEq)]
pub enum ImageEncoding {
    U8,
    U16,
    S16,
    F64
}

impl ImageEncoding {
    // https://en.wikipedia.org/wiki/Normalization_(image_processing)
    //TODO: Make using f64 better
    pub fn normalize_to_gray(&self, max: Float, min : Float, value: Float) -> u8 {
        let range = 255 as Float; // 255 - 0
        ((value - min) * (range / (max - min))) as u8
    }
}