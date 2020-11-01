pub mod sift;
pub mod orb;

#[derive(Debug,Clone)]
pub struct Pyramid<T> {
    pub octaves: Vec<T>
}