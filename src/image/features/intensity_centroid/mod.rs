use crate::{Float,float};
use crate::image::features::geometry::point::Point;
use crate::image::Image;

fn moment((p,q): (u32,u32), image: &Image, points: &Vec<Point<usize>>, center: &Point<usize>) -> Float {
    assert!(p <= 1 && q <= 1);
    let mut moment = 0.0;
    for point in points {
        if point.x < image.buffer.ncols() && point.y < image.buffer.nrows() {
            let intensity = image.buffer[(point.y,point.x)];
            let x = center.x - point.x;
            let y = center.y - point.y;
            moment += ((x.pow(p)*y.pow(q)) as Float)*intensity;
        }
    }

    moment
}


pub fn orientation(image: &Image, points_location: &(Vec<Point<usize>>,Point<usize>)) -> Float {
    let points = &points_location.0;
    let center = points_location.1;
    let m_0_1 = moment((0,1), image, &points, &center);
    let m_1_0 = moment((1,0), image, &points, &center);

    let mut orientation = m_0_1.atan2(m_1_0);
    if orientation < 0.0 {
        orientation += 2.0*float::consts::PI;
    }
    orientation
}

pub fn centroid(image: &Image, points_location: &(Vec<Point<usize>>,Point<usize>)) -> Point<usize> {
    let points = &points_location.0;
    let center = points_location.1;

    let m_0_1 = moment((0,1), image, points, &center);
    let m_1_0 = moment((1,0), image, points, &center);
    let m_0_0 = moment((0,0), image, points, &center);

    let x = m_1_0/m_0_0;
    let y = m_0_1/m_0_0;

    Point { x: x.trunc() as usize, y: y.trunc() as usize}
}