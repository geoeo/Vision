extern crate vision;
extern crate nalgebra as na;

use color_eyre::eyre::Result;
use na::DMatrix;
use vision::{Float,load_runtime_conf};
use vision::io::write_matrix_to_file;


fn main() {
    let runtime_conf = load_runtime_conf();

    let idenitiy = DMatrix::<Float>::identity(5,5);
    write_matrix_to_file(&idenitiy,&runtime_conf.output_path, "test_mat");


}