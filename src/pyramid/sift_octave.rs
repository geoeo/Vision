extern crate image as image_rs;

use crate::image::{Image,image_encoding::ImageEncoding};
use crate::filter::{gauss_kernel::GaussKernel1D, prewitt_kernel::PrewittKernel,laplace_off_center_kernel::LaplaceOffCenterKernel,gaussian_2_d_convolution,filter_1d_convolution};
use crate::{Float,GradientDirection};
use crate::pyramid::runtime_params::RuntimeParams;

#[derive(Debug,Clone)]
pub struct SiftOctave {
    pub images: Vec<Image>,
    pub x_gradient: Vec<Image>,
    pub y_gradient: Vec<Image>,
    pub dog_x_gradient: Vec<Image>,
    pub dog_y_gradient: Vec<Image>,
    pub dog_s_gradient: Vec<Image>,
    pub difference_of_gaussians: Vec<Image>,
    pub sigmas: Vec<Float>
}

impl SiftOctave {

    pub fn build_octave(base_image: &Image, s: usize, sigma_initial: Float, runtime_params: &RuntimeParams) -> SiftOctave {

        let image_count = s + 3;
        let range = 0..image_count;
        let mean = 0.0;
        let blur_width = runtime_params.blur_half_factor;
        let step = 1;

        let prewitt_kernel = PrewittKernel::new();

        //let sigma_0 = (sigma_initial.powi(2) - runtime_params.sigma_in.powi(2)).sqrt()/Octave::inter_pixel_distance(0);
        let sigma_0 = sigma_initial;
        let sigmas: Vec<Float> = range.clone().map(|x| sigma_0*SiftOctave::generate_k(x as Float, s as Float)).collect();
        let kernels: Vec<GaussKernel1D> = sigmas.iter().map(|&sigma| GaussKernel1D::new(mean, sigma,step,SiftOctave::generate_blur_half_width(blur_width,sigma))).collect();
        let images: Vec<Image> = kernels.iter().map(|kernel| gaussian_2_d_convolution(base_image, kernel, false)).collect();
        let images_borrows: Vec<&Image> = images.iter().map(|x| x).collect();
        let x_gradient = range.clone().map(|x| filter_1d_convolution(&images_borrows,x, GradientDirection::HORIZINTAL, &prewitt_kernel, false)).collect();
        let y_gradient = range.clone().map(|x| filter_1d_convolution(&images_borrows,x, GradientDirection::VERTICAL, &prewitt_kernel, false)).collect();


        let mut difference_of_gaussians: Vec<Image> = Vec::with_capacity(image_count-1);
        for i in 0..images.len()-1 {

            let difference_buffer = &images[i+1].buffer - &images[i].buffer;
            difference_of_gaussians.push(Image::from_matrix(&difference_buffer, ImageEncoding::F64, false));
        }
        
        let difference_of_gaussians_borrows: Vec<&Image> = difference_of_gaussians.iter().map(|x| x).collect();
        let dog_range = 0..difference_of_gaussians.len();

        let dog_x_gradient = dog_range.clone().map(|sigma_idx| filter_1d_convolution(&difference_of_gaussians_borrows,sigma_idx, GradientDirection::HORIZINTAL, &prewitt_kernel, false)).collect();
        let dog_y_gradient = dog_range.clone().map(|sigma_idx| filter_1d_convolution(&difference_of_gaussians_borrows,sigma_idx, GradientDirection::VERTICAL, &prewitt_kernel,false )).collect();
        let dog_s_gradient = dog_range.clone().map(|sigma_idx| filter_1d_convolution(&difference_of_gaussians_borrows,sigma_idx, GradientDirection::SIGMA, &prewitt_kernel, false)).collect();
        SiftOctave {images,x_gradient,y_gradient,dog_x_gradient,dog_y_gradient,dog_s_gradient,difference_of_gaussians,sigmas}
    }

    fn generate_k(n: Float, s: Float) -> Float {
        assert!(n >= 0.0);
        let exp = n/s;
        exp.exp2()
    }

    //TODO: check this
    pub fn generate_blur_half_width(blur_half_factor: Float, sigma: Float) -> Float {
        (blur_half_factor*sigma).ceil()
        //blur_half_factor
    }

    pub fn inter_pixel_distance(octave_level: usize) -> Float {
        0.5*(octave_level as Float).exp2()
    }

    pub fn s(&self) -> usize {
        self.sigmas.len() - 3
    }

    pub fn within_range(&self, x: usize, y: usize, sigma_level: usize, kernel_half_width: usize) -> bool {
        let height = self.images[0].buffer.nrows();
        let width = self.images[0].buffer.ncols();
        let sigma_size = self.difference_of_gaussians.len();
        y >= kernel_half_width && y < height-kernel_half_width && 
        x >= kernel_half_width &&  x < width-kernel_half_width &&
        sigma_level >=kernel_half_width && sigma_level < sigma_size-kernel_half_width
    }
}

