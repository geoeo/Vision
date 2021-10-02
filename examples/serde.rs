extern crate nalgebra as na;

use vision::image::features::geometry::point::Point;
use vision::Float;

fn main() -> Result<(), serde_yaml::Error> {

    let point = Point::<Float> { x: 1.0, y: 2.0 };
    let points = vec!(point,point);


    let s = serde_yaml::to_string(&point)?;
    let s_list = serde_yaml::to_string(&points)?;


    println!("{}",s);
    println!("{}",s_list);


    let deserialized_point: Point<Float> = serde_yaml::from_str(&s)?;
    assert_eq!(point, deserialized_point);
    Ok(())

}