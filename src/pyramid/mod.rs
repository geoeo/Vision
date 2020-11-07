pub mod sift;
pub mod orb;

#[derive(Debug,Clone)]
pub struct Pyramid<T> {
    pub octaves: Vec<T>
}

impl<T> Pyramid<T> {
    pub fn empty(number_of_octave: usize) -> Pyramid<T> {
        Pyramid::<T> {octaves: Vec::<T>::with_capacity(number_of_octave)}
    }
}