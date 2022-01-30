extern crate vision;
extern crate nalgebra as na;

use vision::Float;
use vision::numerics::linear_prog::l1_norm_approx;
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
for i in 0..q.nrows() {
    let v_norm: Float = (q[i] as Float)/(q.max() as Float);
    q[(i,0)] = ((M as Float)*v_norm).round() as usize;
}
let mut y = c.clone();
for i in 0..T {
    let v = q[i];
    y[v] = random[(i,0)];
}

//recover
let R = G_t*G;
let x0 = R.cholesky().expect("l1 decode example cholesky failed").solve(&(G_t*y));
//let xp = l1_norm_approx(&y,&G,&x0, 200, 1e-4);


}
