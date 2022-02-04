extern crate vision;
extern crate nalgebra as na;

use vision::Float;
use vision::numerics::linear_prog::l1_norm_approx;
use vision::io::octave_loader::{load_matrix,load_vector};
use na::{DMatrix, DVector};


fn main() {

let mut x = load_vector("D:/Workspace/Datasets/Octave/x0_small.txt");
let y = load_vector("D:/Workspace/Datasets/Octave/y_small.txt");
let G = load_matrix("D:/Workspace/Datasets/Octave/G_small.txt");

// println!("{}",x);
// println!("-----");
// println!("{}",y);
// println!("-----");
// println!("{}",G);

l1_norm_approx(&y,&G,&mut x, 200, 1e-4);

println!("{}",x);


}
