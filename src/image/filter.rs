use crate::image::{Image,gauss_kernel::GaussKernel, kernel::Kernel};
use crate::pyramid::octave::Octave;
use crate::{Float, GradientDirection};
use crate::extrema::extrema_parameters::ExtremaParameters;

pub fn filter_1d_convolution(source_images: &Vec<&Image>, sigma_level: usize, filter_direction: GradientDirection, filter_kernel: &dyn Kernel, normalize: bool) -> Image {
    let kernel = &filter_kernel.kernel();
    let step = filter_kernel.step();
    let kernel_half_width = filter_kernel.half_width();
    let kernel_half_width_signed = kernel_half_width as isize;
    
    let source = &source_images[sigma_level];
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();
    let mut target =  Image::empty(height, width, source.original_encoding);

    for y in 0..height {
        for x in 0..width {
                let mut acc = 0.0;
                for kenel_idx in (-kernel_half_width_signed..kernel_half_width_signed+1).step_by(step){

                    let sample_value = match filter_direction {
                        GradientDirection::HORIZINTAL => {
                            let sample_idx = (x as isize)+kenel_idx;
                            match sample_idx {
                                sample_idx if sample_idx < 0 =>  buffer.index((y,0)),
                                sample_idx if sample_idx >= width as isize => buffer.index((y,width -1)),
                                _ => buffer.index((y,sample_idx as usize))
                            }
                        },
                        GradientDirection::VERTICAL => {
                            let sample_idx = (y as isize)+kenel_idx;
                            match sample_idx {
                                sample_idx if sample_idx < 0 =>  buffer.index((0,x)),
                                sample_idx if sample_idx >= height as isize => buffer.index((height -1,x)),
                                _ =>  buffer.index((sample_idx as usize, x))
                            }
                        },
                        GradientDirection::SIGMA => {
                            let sample_idx = sigma_level as isize + kenel_idx;
                            let sample_buffer = match sample_idx {
                                sample_idx if sample_idx < 0 =>  &source_images[0].buffer,
                                sample_idx if sample_idx >= source_images.len() as isize => &source_images[source_images.len()-1].buffer,
                                _ => &source_images[sample_idx as usize].buffer
                                
                            };

                            sample_buffer.index((y,x))

                        }
                    };

                
                    let kenel_value = kernel[(kenel_idx + kernel_half_width_signed) as usize];
                    acc +=sample_value*kenel_value;
                }
                target.buffer[(y,x)] = acc/filter_kernel.normalizing_constant(); //TODO: this is not correct for all filters
        }
    }

    if normalize {
        target.buffer.normalize_mut();
    }
    target
}

//TODO: improve this
pub fn gradient_convolution_at_sample(source_octave: &Octave,source_images: &Vec<Image>, input_params: &ExtremaParameters, filter_kernel: &dyn Kernel, gradient_direction: GradientDirection) -> Float {

    let x_input = input_params.x; 
    let x_input_signed = input_params.x as isize; 
    let y_input = input_params.y; 
    let y_input_signed = input_params.y as isize; 
    let sigma_level_input = input_params.sigma_level;
    let sigma_level_input_signed = input_params.sigma_level as isize;

    let kernel = filter_kernel.kernel();
    let step = filter_kernel.step();
    let repeat = filter_kernel.half_repeat() as isize;
    let repeat_range = -repeat..repeat+1;
    let kernel_half_width = filter_kernel.half_width();
    let kernel_half_width_signed = kernel_half_width as isize;

    let buffer = &source_images[input_params.sigma_level].buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();


    match gradient_direction {
        GradientDirection::HORIZINTAL => {
            assert!(x_input_signed -kernel_half_width_signed >= 0 && x_input + kernel_half_width <= width);
            assert!(x_input_signed -repeat >= 0 && x_input + (repeat as usize) < width);
         },
        GradientDirection::VERTICAL => { 
            assert!(y_input_signed -kernel_half_width_signed >= 0 && y_input + kernel_half_width <= height);
            assert!(y_input_signed -repeat >= 0 && y_input + (repeat as usize) < height);
         },
        GradientDirection::SIGMA => { 
            assert!(sigma_level_input_signed -kernel_half_width_signed >= 0 && sigma_level_input + kernel_half_width < source_octave.difference_of_gaussians.len());
        }

    }

    let mut convolved_value = 0.0;
    for kenel_idx in (-kernel_half_width_signed..kernel_half_width_signed+1).step_by(step){
        let kenel_value = kernel[(kenel_idx + kernel_half_width_signed) as usize];

            let weighted_sample = match gradient_direction {
                GradientDirection::HORIZINTAL => {
                    let mut acc = 0.0;
                    for repeat_idx in repeat_range.clone() {
                        let sample_idx = x_input_signed +kenel_idx;
                        let y_repeat_idx =  y_input_signed + repeat_idx;

                        let sample_value = buffer.index((y_repeat_idx as usize,sample_idx as usize));
                        acc += kenel_value*sample_value;
                    }
                    let range_size = repeat_range.end - repeat_range.start;
                    acc/ (range_size as Float * filter_kernel.normalizing_constant())
                },
                GradientDirection::VERTICAL => {
                    let mut acc = 0.0;
                    for repeat_idx in repeat_range.clone() {
                        let sample_idx = y_input_signed+kenel_idx;
                        let x_repeat_idx = x_input_signed + repeat_idx;
                        let sample_value = buffer.index((sample_idx as usize, x_repeat_idx as usize));
                        acc += kenel_value*sample_value;
                    }
                    let range_size = repeat_range.end - repeat_range.start;
                    acc/ (range_size as Float * filter_kernel.normalizing_constant())
                },
                GradientDirection::SIGMA => { 
                    let mut acc = 0.0;
                    //for repeat_idx in repeat_range.clone() {
                    let sigma_range = 0..1;
                    for repeat_idx in sigma_range {
                        let sample_idx = sigma_level_input_signed + kenel_idx;
                        let x_repeat_idx = x_input_signed + repeat_idx;
                        let sample_buffer =  &source_images[sample_idx as usize].buffer;
                        let sample_value = sample_buffer.index((y_input,x_repeat_idx as usize));
                        acc += kenel_value*sample_value;
                    }
                    let range_size = repeat_range.end - repeat_range.start;
                    acc/ (range_size as Float * filter_kernel.normalizing_constant())
                    //sample_value*kenel_value
                }
                
            };

            convolved_value += weighted_sample;
    
    }

    convolved_value
}
 


pub fn gaussian_2_d_convolution(image: &Image, filter_kernel: &GaussKernel, normalize: bool) -> Image {
    let vec = vec![image];
    let blur_hor = filter_1d_convolution(&vec,0,GradientDirection::HORIZINTAL, filter_kernel, normalize);
    let vec_2 = vec![&blur_hor];
    filter_1d_convolution(&vec_2,0,GradientDirection::VERTICAL, filter_kernel, normalize)
}

