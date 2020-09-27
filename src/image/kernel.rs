use crate::Float;

pub trait Kernel {
    // Filter
    fn kernel(&self) -> &Vec<Float>;
    // Size at which the filter is traversed
    fn step(&self) -> usize;
    // Half of the width of the kernel save the center element
    fn half_width(&self) -> usize {
        (self.kernel().len()-1)/2
    }
    // Half the number of extra repeats of the kernel
    fn half_repeat(&self) -> usize;

    fn normalizing_constant(&self) -> Float;
}
