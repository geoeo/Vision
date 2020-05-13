use std::f64::consts::PI;
use crate::image::Image;

fn gaussian_sample(mean: f64, std: f64, x:f64) -> f64 {
    let exponent = (-0.5*((x-mean)/std).powi(2)).exp();
    let factor = 1.0/(std*(2.0*PI).sqrt());
    factor*exponent
}

fn gaussian_1_d_kernel(mean: f64, std: f64, step: i8, end: i8) -> Vec<f64> {
    assert_eq!(end%step,0);
    assert!(end > 0);

    let range = (-end..end+1).step_by(step as usize);
    range.map(|x| gaussian_sample(mean,std,x as f64)).collect()
}

//TODO: normalize both

pub fn gaussian_1_d_convolution_horizontal(source: & Image, target: &mut Image ,mean: f64, std: f64) -> () {
    let step: i8 = 1;
    let end: i8 = 3;
    let offset = (end as usize)/(step as usize);
    let offset_signed = offset as i32;

    let kernel = gaussian_1_d_kernel(mean, std, step, end);
    let buffer = &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();


    for y in 0..height {
        for x in 0..width {
                let mut acc = 0.0;
                for i in (-offset_signed..offset_signed+1).step_by(step as usize){
                    let sample_idx = (x as i32)+i;
                    let kenel_idx = i +offset as i32;

                    let sample_value = match x {
                        x if x < offset =>  buffer.index((y,0)),
                        x if x >=  width-offset => buffer.index((y,width -1)),
                        _ => buffer.index((y,sample_idx as usize))
                    };
                    let kenel_value = kernel[kenel_idx as usize];
                    acc +=sample_value*kenel_value;
                }

                target.buffer[(y,x)] = acc;

        }



    }





}

pub fn gaussian_1_d_convolution_vertical(source: & Image, target: &mut Image,mean: f64, std: f64) -> () {
    let step: i8 = 1;
    let end: i8 = 3;
    let offset = (end as usize)/(step as usize);
    let offset_signed = offset as i32;

    let kernel = gaussian_1_d_kernel(mean, std, step, end);
    let buffer =  &source.buffer;
    let width = buffer.ncols();
    let height = buffer.nrows();

    for x in 0..width {
        for y in 0..height {

            let mut acc = 0.0;
            for i in (-offset_signed..offset_signed+1). step_by(step as usize){
                let sample_idx = (y as i32)+i;
                let kenel_idx = i +offset as i32;
                let sample_value = match y {
                    y if y < offset =>  buffer.index((0,x)),
                    y if y >=  height-offset => buffer.index((height -1,x)),
                    _ =>  buffer.index((sample_idx as usize, x))
                };
                let kenel_value = kernel[kenel_idx as usize];
                acc +=sample_value*kenel_value;
            }

            target.buffer[(y,x)] = acc;
        }

    }

}

