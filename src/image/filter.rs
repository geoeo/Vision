use crate::image::{Image,gauss_kernel::GaussKernel, kernel::Kernel};

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum FilterDirection {
    HORIZINTAL,
    VERTICAL
}

pub fn gaussian_1_d_convolution(source: &Image, filter_direction: FilterDirection, filter_kernel: &dyn Kernel) -> Image {
    let kernel = filter_kernel.kernel();
    let step = filter_kernel.step();
    let kernel_half_width = filter_kernel.half_width();
    let kernel_half_width_signed = kernel_half_width as isize;
    
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();
    let mut target =  Image::empty(height, width, source.original_encoding);


    for y in 0..height {
        for x in 0..width {
                let mut acc = 0.0;
                for kenel_idx in (-kernel_half_width_signed..kernel_half_width_signed+1).step_by(step){


                    //TODO: Allow for applying the filter to consequtive rows/columns
                    let sample_value = match filter_direction {
                        FilterDirection::HORIZINTAL => {
                            let sample_idx = (x as isize)+kenel_idx;
                            match sample_idx {
                                sample_idx if sample_idx < 0 =>  buffer.index((y,0)),
                                sample_idx if sample_idx >=  (width-kernel_half_width) as isize => buffer.index((y,width -1)),
                                _ => buffer.index((y,sample_idx as usize))
                            }
                        },
                        FilterDirection::VERTICAL => {
                            let sample_idx = (y as isize)+kenel_idx;
                            match sample_idx {
                                sample_idx if sample_idx < 0 =>  buffer.index((0,x)),
                                sample_idx if sample_idx >=  (height-kernel_half_width) as isize => buffer.index((height -1,x)),
                                _ =>  buffer.index((sample_idx as usize, x))
                            }
                        }
                    };

                
                    let kenel_value = kernel[(kenel_idx + kernel_half_width_signed) as usize];
                    acc +=sample_value*kenel_value;
                }
                target.buffer[(y,x)] = acc;
        }
    }

    target

}


pub fn gaussian_2_d_convolution(image: &Image, filter_kernel: &GaussKernel) -> Image {
    let blur_hor = gaussian_1_d_convolution(image,FilterDirection::HORIZINTAL, filter_kernel);
    gaussian_1_d_convolution(&blur_hor,FilterDirection::VERTICAL, filter_kernel)
}

