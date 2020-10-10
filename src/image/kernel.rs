extern crate nalgebra as na;

use na::DMatrix;
use crate::Float;

pub trait Kernel{
    // Filter
    fn kernel(&self) -> &DMatrix<Float>;
    // Size at which the filter is traversed
    fn step(&self) -> usize;
    // Half of the width of the kernel save the center element
    fn half_width(&self) -> usize {
        (self.kernel().ncols()-1)/2
    }
    // Half the number of extra repeats of the kernel -> TODO: Make this dervied from Matrix height
    fn half_repeat(&self) -> usize {
        (self.kernel().nrows()-1)/2
    }
    fn normalizing_constant(&self) -> Float;
}
