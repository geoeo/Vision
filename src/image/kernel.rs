use crate::Float;

pub trait Kernel {
    // Filter
    fn kernel(&self) -> &Vec<Float>;
    // Size at which the filter is traversed
    fn step(&self) -> usize;
    // Half of the width of the kernel
    fn half_width(&self) -> usize {
        (self.kernel().len()-1)/2
    }
}
