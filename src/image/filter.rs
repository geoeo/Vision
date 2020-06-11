use crate::image::{Image,gauss_kernel::GaussKernel, kernel::Kernel};

#[repr(u8)]
#[derive(Debug,Copy,Clone)]
pub enum FilterDirection2D {
    HORIZINTAL,
    VERTICAL
}

pub fn filter_1d_convolution(source: &Image, filter_direction: FilterDirection2D, filter_kernel: &dyn Kernel) -> Image {
    let kernel = &filter_kernel.kernel();
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


                    let sample_value = match filter_direction {
                        FilterDirection2D::HORIZINTAL => {
                            let sample_idx = (x as isize)+kenel_idx;
                            match sample_idx {
                                sample_idx if sample_idx < 0 =>  buffer.index((y,0)),
                                sample_idx if sample_idx >=  (width-kernel_half_width) as isize => buffer.index((y,width -1)),
                                _ => buffer.index((y,sample_idx as usize))
                            }
                        },
                        FilterDirection2D::VERTICAL => {
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
    let blur_hor = filter_1d_convolution(image,FilterDirection2D::HORIZINTAL, filter_kernel);
    filter_1d_convolution(&blur_hor,FilterDirection2D::VERTICAL, filter_kernel)
}

