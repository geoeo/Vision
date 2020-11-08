use crate::features::geometry::point::Point;

#[derive(Debug,Clone)]
pub struct Line<T> where T: PartialOrd + PartialEq {
    pub points: Vec<Point<T>>
}

// https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#:~:text=Bresenham%27s%20line%20algorithm%20is%20a,straight%20line%20between%20two%20points.
pub fn line_bresenham(point_a: &Point<usize>, point_b: &Point<usize>) -> Line<usize> {

/*

plotLine(int x0, int y0, int x1, int y1)
    dx =  abs(x1-x0);
    sx = x0<x1 ? 1 : -1;
    dy = -abs(y1-y0);
    sy = y0<y1 ? 1 : -1;
    err = dx+dy;  /* error value e_xy */
    while (true)   /* loop */
        plot(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2*err;
        if (e2 >= dy) /* e_xy+e_x > 0 */
            err += dy;
            x0 += sx;
        end if
        if (e2 <= dx) /* e_xy+e_y < 0 */
            err += dx;
            y0 += sy;
        end if
    end while

*/  

    panic!("Not implemented yet");

}