extern crate vision;
extern crate nalgebra as na;

use vision::Float;
use vision::numerics::linear_prog::l1_norm_approx;
use vision::io::octave_loader::{load_matrix,load_vector};
use na::{DMatrix, DVector};


fn main() {

//source length
let N = 256;

//codeword length
let M = 4*N;

//number of perturbations
let T = (0.2*(M as Float)).round() as usize;

//coding matrix
let G = DMatrix::<Float>::new_random(M,N);
let G_t = &G.transpose();

//source word
let x = DVector::<Float>::new_random(N);

//code word
let c = &G*x;


//channel: perturb T randomly chosen entries
let mut q = DVector::<usize>::new_random(M);
let random = DVector::<Float>::new_random(T);
for i in 0..M {
    let v_norm: Float = (q[i] as Float)/(q.max() as Float);
    q[i] = ((N as Float)*v_norm).round() as usize;
}
let mut y = c.clone();
for i in 0..T {
    let v = q[i];
    y[v] = random[i];
}

//recover
let R = G_t*(&G);
//let mut x = R.cholesky().expect("l1 decode example cholesky failed").solve(&(G_t*(&y)));

let mut x = load_vector("D:/Workspace/Datasets/Octave/x0.txt");
let y = load_vector("D:/Workspace/Datasets/Octave/y.txt");
let G = load_matrix("D:/Workspace/Datasets/Octave/G.txt");

//l1_norm_approx(&y,&G,&mut x, 200, 1e-4);


}
