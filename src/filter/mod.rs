use crate::image::Image;
use crate::{Float, GradientDirection};
use crate::features::Feature;
use self::{kernel::Kernel,gauss_kernel::GaussKernel1D};

pub mod gauss_kernel;
pub mod prewitt_kernel;
pub mod laplace_kernel;
pub mod laplace_off_center_kernel;
pub mod kernel;


pub fn filter_1d_convolution(source_images: &Vec<&Image>, sigma_level: usize, filter_direction: GradientDirection, filter_kernel: &dyn Kernel, normalize: bool) -> Image {
    let kernel = &filter_kernel.kernel();
    let step = filter_kernel.step();
    let kernel_radius = filter_kernel.radius();
    let kernel_radius_signed = kernel_radius as isize;
    
    let source = &source_images[sigma_level];
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();
    let mut target =  Image::empty(width, height, source.original_encoding);

    for y in 0..height {
        for x in 0..width {
                let mut acc = 0.0;
                for kenel_idx in (-kernel_radius_signed..kernel_radius_signed+1).step_by(step){

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

                
                    let kenel_value = kernel[(0,(kenel_idx + kernel_radius_signed) as usize)];
                    acc +=sample_value*kenel_value;
                }

                target.buffer[(y,x)] = acc/filter_kernel.normalizing_constant(); 
        }
    }

    if normalize {
        target.buffer.normalize_mut();
    }
    target
}

//TODO: performance
pub fn gradient_convolution_at_sample(source_images: &Vec<Image>,input_params: &dyn Feature, filter_kernel: &dyn Kernel, gradient_direction: GradientDirection) -> Float {

    let x_input = input_params.get_x_image(); 
    let x_input_signed = x_input as isize; 
    let y_input = input_params.get_y_image(); 
    let y_input_signed = y_input as isize; 
    let sigma_level_input = input_params.get_closest_sigma_level();
    let sigma_level_input_signed = sigma_level_input as isize;

    let kernel = filter_kernel.kernel();
    let step = filter_kernel.step();
    let kernel_radius = filter_kernel.radius();
    let kernel_radius_signed = kernel_radius as isize;

    let buffer = &source_images[sigma_level_input].buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();


    match gradient_direction {
        GradientDirection::HORIZINTAL => {
            if !(x_input_signed -kernel_radius_signed >= 0 && x_input + kernel_radius < width){
                panic!("HORIZONTAL: cant convolve at location: {} with radius: {}, image width: {}",x_input_signed, kernel_radius_signed, width);
            }
         },
        GradientDirection::VERTICAL => { 
            if !(y_input_signed -kernel_radius_signed >= 0 && y_input + kernel_radius < height){
                panic!("VERTICAL: cant convolve at location: {} with radius: {} image, height: {}",y_input_signed, kernel_radius_signed, height);
            }
         },
        GradientDirection::SIGMA => { 
            if !(sigma_level_input_signed -kernel_radius_signed >= 0 && sigma_level_input + kernel_radius < source_images.len()) {
                panic!("SIGMA: cant convolve at location: {} with radius: {}",sigma_level_input_signed, kernel_radius_signed);
            }
        }

    }

    //TODO: kernel repeats
    let mut convolved_value = 0.0;
    for kenel_idx in (-kernel_radius_signed..kernel_radius_signed+1).step_by(step){
        let kenel_value = kernel[(kenel_idx + kernel_radius_signed) as usize];

            let weighted_sample = match gradient_direction {
                GradientDirection::HORIZINTAL => {
                    let mut acc = 0.0;
                    let sample_idx = x_input_signed +kenel_idx;
                    let sample_value = buffer.index((y_input ,sample_idx as usize));
                    acc += kenel_value*sample_value;
                    acc/ filter_kernel.normalizing_constant()
                },
                GradientDirection::VERTICAL => {
                    let mut acc = 0.0;
                    let sample_idx = y_input_signed+kenel_idx;
                    let sample_value = buffer.index((sample_idx as usize, x_input));
                    acc += kenel_value*sample_value;
                    acc/ filter_kernel.normalizing_constant()
                },
                GradientDirection::SIGMA => { 
                    let mut acc = 0.0;
                    let sample_idx = sigma_level_input_signed + kenel_idx;
                    let sample_buffer =  &source_images[sample_idx as usize].buffer;
                    let sample_value = sample_buffer.index((y_input,x_input));
                    acc += kenel_value*sample_value;

                    acc/ filter_kernel.normalizing_constant()
                }
                
            };

            convolved_value += weighted_sample;
    
    }

    convolved_value
}
 


pub fn gaussian_2_d_convolution(image: &Image, filter_kernel: &GaussKernel1D, normalize: bool) -> Image {
    let vec = vec![image];
    let blur_hor = filter_1d_convolution(&vec,0,GradientDirection::HORIZINTAL, filter_kernel, false);
    let vec_2 = vec![&blur_hor];
    filter_1d_convolution(&vec_2,0,GradientDirection::VERTICAL, filter_kernel, normalize)
}

