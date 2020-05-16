use crate::{Float,float};
use float::consts::PI;

use crate::image::Image;

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum FilterDirection {
    HORIZINTAL,
    VERTICAL
}


fn gaussian_sample(mean: Float, std: Float, x:Float) -> Float {
    let exponent = (-0.5*((x-mean)/std).powi(2)).exp();
    let factor = 1.0/(std*(2.0*PI).sqrt());
    factor*exponent
}

fn gaussian_1_d_kernel(mean: Float, std: Float, step: i8, end: i8) -> Vec<Float> {
    assert_eq!(end%step,0);
    assert!(end > 0);

    let range = (-end..end+1).step_by(step as usize);
    range.map(|x| gaussian_sample(mean,std,x as Float)).collect()
}

pub fn gaussian_1_d_convolution(source: &Image, filter_direction: FilterDirection, mean: Float, std: Float, step: i8, end: i8) -> Image {
    let kernel = gaussian_1_d_kernel(mean, std, step, end);
    
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();
    let mut target =  Image::empty(height, width, source.original_encoding);


    for y in 0..height {
        for x in 0..width {
                let mut acc = 0.0;
                for kenel_idx in (-end..end+1).step_by(step as usize){


                    let sample_value = match filter_direction {
                        FilterDirection::HORIZINTAL => {
                            let sample_idx = (x as i32)+(kenel_idx as i32);
                            match sample_idx {
                                sample_idx if sample_idx < 0 =>  buffer.index((y,0)),
                                sample_idx if sample_idx >=  (width-end as usize) as i32 => buffer.index((y,width -1)),
                                _ => buffer.index((y,sample_idx as usize))
                            }
                        },
                        FilterDirection::VERTICAL => {
                            let sample_idx = (y as i32)+(kenel_idx as i32);
                            match sample_idx {
                                sample_idx if sample_idx < 0 =>  buffer.index((0,x)),
                                sample_idx if sample_idx >=  (height-end as usize) as i32 => buffer.index((height -1,x)),
                                _ =>  buffer.index((sample_idx as usize, x))
                            }
                        }
                    };

                
                    let kenel_value = kernel[(kenel_idx + end) as usize];
                    acc +=sample_value*kenel_value;
                }
                target.buffer[(y,x)] = acc;
        }
    }

    target

}


pub fn gaussian_2_d_convolution(image: &Image, mean:Float, sigma: Float, step: i8, end: i8) -> Image {
    let blur_hor = gaussian_1_d_convolution(image,FilterDirection::HORIZINTAL, mean, sigma, step, end);
    gaussian_1_d_convolution(&blur_hor,FilterDirection::VERTICAL, mean, sigma, step, end)
}

