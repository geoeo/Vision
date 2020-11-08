use crate::features::geometry::point::Point;

#[derive(Debug,Clone)]
pub struct Line<T> where T: PartialOrd + PartialEq {
    pub points: Vec<Point<T>>
}

// https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#:~:text=Bresenham%27s%20line%20algorithm%20is%20a,straight%20line%20between%20two%20points.
pub fn line_bresenham(point_a: &Point<usize>, point_b: &Point<usize>) -> Line<usize> {
    let mut points = Vec::<Point<usize>>::new();

    let mut x0 = point_a.x as isize;
    let mut y0 = point_a.y as isize;
    let x1 = point_b.x as isize;
    let y1 = point_b.y as isize;

    let dx = (x1 -x0).abs();
    let sx = if x0 < x1 {1} else {-1};
    let dy = -(y1-y0).abs();
    let sy = if y0 < y1 {1} else {-1};
    let mut err = dx + dy;
    loop {
        assert!(x0 >= 0 && y0 >= 0);
        points.push(Point{x: x0 as usize, y: y0 as usize});
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2*err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }

    Line::<usize> {points}

}