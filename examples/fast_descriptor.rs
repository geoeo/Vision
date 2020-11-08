extern crate image as image_rs;
extern crate vision;

use std::path::Path;

use vision::image::Image;
use vision::visualize::draw_points;
use vision::features::geometry::shape::Shape;
use vision::features::fast_feature::FastFeature;

fn main() {
    let image_name = "lenna";
    let image_format = "png";
    let image_path = format!("images/{}.{}",image_name, image_format);
    let converted_file_out_path = format!("output/{}_fast.{}",image_name,image_format);

    println!("Processing Image: {}", image_name);

    let image = image_rs::open(&Path::new(&image_path)).unwrap().to_luma();
    let mut frame = Image::from_gray_image(&image, false);


    let valid_features = FastFeature::compute_valid_features(&frame,3,0.2,12,(10,10));
    println!("amount of FAST features:{:?}",valid_features.len());
    for (valid_feature,i) in valid_features {
        let slice = valid_feature.get_wrapping_slice(i, 12);
        let points = Shape::points(valid_feature.location.x, valid_feature.location.y, &slice);
        let full_circle = valid_feature.get_full_circle();
        draw_points(&mut frame, &points, 64.0);
        //draw_points(&mut frame, &full_circle.shape.get_points(), 64.0);
    }





    let new_image = frame.to_image();

    new_image.save(converted_file_out_path).unwrap();

}