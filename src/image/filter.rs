use crate::image::{Image,gauss_kernel::GaussKernel, kernel::Kernel};
use crate::Float;

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum FilterDirection {
    HORIZINTAL,
    VERTICAL
}

pub fn filter_convolution(source: &Image, filter_direction: FilterDirection, filter_kernel: &dyn Kernel) -> Image {
    let kernel = filter_kernel.kernel();
    let step = filter_kernel.step();
    let kernel_half_width = filter_kernel.half_width();
    let repeat = filter_kernel.half_repeat() as isize;
    let repeat_range = -repeat..repeat+1;
    let kernel_half_width_signed = kernel_half_width as isize;
    
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();
    let mut target =  Image::empty(height, width, source.original_encoding);


    for y in 0..height {
        for x in 0..width {
                let mut acc : Float = 0.0;
                for kenel_idx in (-kernel_half_width_signed..kernel_half_width_signed+1).step_by(step){

                    //TODO: Repeat > 0 doesnt seem to work

                    for repeat_idx in repeat_range.clone() {
                        let sample_value = match filter_direction {
                            FilterDirection::HORIZINTAL => {
                                let sample_idx = (x as isize)+kenel_idx;
                                let y_repeat_idx =  match y as isize + repeat_idx {
                                    y_idx if y_idx < 0 => 0,
                                    y_idx if y_idx >= height as isize => height-1,
                                    y_idx => y_idx as usize
                                };
                                
                                match sample_idx {
                                    sample_idx if sample_idx < 0 =>  buffer.index((y_repeat_idx,0)),
                                    sample_idx if sample_idx >=  (width-kernel_half_width) as isize => buffer.index((y_repeat_idx,width -1)),
                                    _ => buffer.index((y_repeat_idx,sample_idx as usize))
                                }
                            },
                            FilterDirection::VERTICAL => {
                                let sample_idx = (y as isize)+kenel_idx;
                                let x_repeat_idx = match x as isize + repeat_idx {
                                    x_idx if x_idx < 0 => 0,
                                    x_idx if x_idx >= width as isize => width-1,
                                    x_idx => x_idx as usize
                                };

                                match sample_idx {
                                    sample_idx if sample_idx < 0 =>  buffer.index((0,x_repeat_idx)),
                                    sample_idx if sample_idx >=  (height-kernel_half_width) as isize => buffer.index((height -1,x_repeat_idx)),
                                    _ =>  buffer.index((sample_idx as usize, x_repeat_idx))
                                }
                            }
                        };
                        let kenel_value = kernel[(kenel_idx + kernel_half_width_signed) as usize];
                        acc +=sample_value*kenel_value;
                    }   
                }
                let range_size = repeat_range.end - repeat_range.start;
                acc /= range_size as Float;
                target.buffer[(y,x)] = acc;
        }
    }

    target

}


pub fn gaussian_2_d_convolution(image: &Image, filter_kernel: &GaussKernel) -> Image {
    let blur_hor = filter_convolution(image,FilterDirection::HORIZINTAL, filter_kernel);
    filter_convolution(&blur_hor,FilterDirection::VERTICAL, filter_kernel)
}

