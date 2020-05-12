use std::f64::consts::PI;
use crate::image::Image;

fn gaussian_sample(mean: f64, std: f64, x:f64) -> f64 {
    let exponent = ((x-mean)/(-2.0*std)).powi(2).exp();
    let factor = 1.0/(std*(2.0*PI).sqrt());
    factor*exponent
}

fn gaussian_1_d_kernel(mean: f64, std: f64, step: i8, end: i8) -> Vec<f64> {
    assert_eq!(end%step,0);
    assert!(end > 0);

    let range = (-end..end+1).step_by(step as usize);
    range.map(|x| gaussian_sample(mean,std,x as f64)).collect()
}

pub fn gaussian_1_d_convolution_horizontal(image: &mut Image,mean: f64, std: f64) -> () {
    let step: i8 = 1;
    let end: i8 = 2;
    let offset = (end as usize)/(step as usize);
    let offset_signed = offset as i32;

    let kernel = gaussian_1_d_kernel(mean, std, step, end);
    let buffer = &mut image.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();

    for y in 0..height {
        for x in offset..width-offset {
            let mut acc = 0.0;
            for i in (-offset_signed..offset_signed+1). step_by(step as usize){
                let sample_idx = (x as i32)+i;
                let kenel_idx = i as usize +offset;
                let sample_value = buffer.index((y,sample_idx as usize));
                let kenel_value = kernel[kenel_idx];
                acc +=sample_value*kenel_value;
            }
            buffer[(y,x)] = acc;
        }

    }


}

