extern crate nalgebra as na;

use na::{DVector,DMatrix};
use std::fs::File;
use std::io::{BufReader,BufRead, Lines};
use crate::io::{load_image_as_gray,parse_to_float, octave_loader::{load_matrices,load_matrix,load_vector}};
use crate::image::Image;

use crate::Float;

