use crate::{Float,float};
use crate::features::geometry::point::Point;
use crate::image::Image;

fn moment((p,q): (u32,u32), image: &Image, points: &Vec<Point<usize>>) -> Float {
    assert!(p <= 1 && q <= 1);
    let mut moment = 0.0;
    for point in points {
        let intensity = image.buffer[(point.y,point.x)];
        moment += ((point.x.pow(p)*point.y.pow(q)) as Float)*intensity;
    }

    moment
}

pub fn orientation(image: &Image, points: &Vec<Point<usize>>) -> Float {
    let m_0_1 = moment((0,1), image, points);
    let m_1_0 = moment((1,0), image, points);

    let mut orientation = m_0_1.atan2(m_1_0);
    if orientation < 0.0 {
        orientation += 2.0*float::consts::PI;
    }
    orientation
}

pub fn centroid(image: &Image, points: &Vec<Point<usize>>) -> Point<usize> {
    let m_0_1 = moment((0,1), image, points);
    let m_1_0 = moment((1,0), image, points);
    let m_0_0 = moment((0,0), image, points);

    let x = m_1_0/m_0_0;
    let y = m_0_1/m_0_0;

    Point { x: x.trunc() as usize, y: y.trunc() as usize}
}