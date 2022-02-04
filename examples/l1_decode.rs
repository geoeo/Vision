extern crate vision;
extern crate nalgebra as na;

use vision::numerics::linear_prog::l1_norm_approx;
use vision::io::octave_loader::{load_matrix,load_vector};


fn main() {

let mut x = load_vector("D:/Workspace/Datasets/Octave/x0.txt");
let y = load_vector("D:/Workspace/Datasets/Octave/y.txt");
let G = load_matrix("D:/Workspace/Datasets/Octave/G.txt");

l1_norm_approx(&y,&G,&mut x, 200, 1e-4);

println!("{}",x);


}
