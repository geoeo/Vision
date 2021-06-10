extern crate nalgebra as na;

use na::Matrix2;
use crate::image::Image;
use crate::{Float, GradientDirection};
use crate::filter::{gradient_convolution_at_sample,prewitt_kernel::PrewittKernel};
use crate::features::{Feature,orb_feature::OrbFeature};
use crate::features::geometry::point::Point;

pub fn harris_matrix(images: &Vec<Image>, feature: &dyn Feature, window_size: usize) -> Matrix2<Float> {
    assert_eq!(images.len(),1);
    let first_order_kernel = PrewittKernel::new();


    let mut dx_acc: Float = 0.0;
    let mut dy_acc: Float = 0.0;
    let mut dxdy_acc: Float = 0.0;

    let r = (window_size -1 /2) as isize;

    let pos_x = feature.get_x_image() as isize;
    let pos_y = feature.get_y_image() as isize;

    for j in -r..r {
        for i in -r..r {
            let sample_x = match pos_x+ i {
                v if v < 0 => 0 as usize,
                v if v > (images[0].buffer.ncols()-1) as isize => images[0].buffer.ncols()-1  as usize,
                v => v as usize
            };
            let sample_y = match pos_y+ j {
                v if v < 0 => 0 as usize,
                v if v > (images[0].buffer.nrows()-1) as isize => images[0].buffer.nrows()-1  as usize,
                v => v as usize
            };
            
            //TODO: orientaiton not needed here
            let sample_feature = OrbFeature { location: Point {x:sample_x, y: sample_y }, orientation: 0.0 };

            //TODO: this may crash if window parameters make it go out of bounds
            let dx = gradient_convolution_at_sample(images,&sample_feature,&first_order_kernel,GradientDirection::HORIZINTAL);
            let dy =  gradient_convolution_at_sample(images,&sample_feature,&first_order_kernel,GradientDirection::VERTICAL);
            let dxdy = dx*dy;

            dx_acc += dx;
            dy_acc += dy;
            dxdy_acc += dxdy; 


        }
    }

    Matrix2::new(dx_acc.powi(2),dxdy_acc,
                 dxdy_acc,dy_acc.powi(2))

    //let dx = gradient_convolution_at_sample(images,feature,&first_order_kernel,GradientDirection::HORIZINTAL);
    //let dy = gradient_convolution_at_sample(images,feature,&first_order_kernel,GradientDirection::VERTICAL);

    // Matrix2::new(dx.powi(2),dx*dy,
    //             dx*dy,dy.powi(2))

}

pub fn harris_response(harris_matrix: &Matrix2<Float>, k: Float) -> Float {
    let determinant = harris_matrix.determinant();
    let trace = harris_matrix.trace();
    determinant - k*trace.powi(2)
}

//TODO: this either pass in orientation or store it in the feature (actually not needed only to satisfy Orb creation)
pub fn harris_response_for_feature(images: &Vec<Image>, feature: &dyn Feature,  k: Float, window_size: usize) -> Float {
    let harris_matrix = harris_matrix(images,feature,window_size);
    harris_response(&harris_matrix, k)
}
