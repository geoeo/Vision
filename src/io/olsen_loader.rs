extern crate nalgebra as na;

use na::DMatrix;
use crate::io::{ octave_loader::{load_matrices,load_matrix},load_images};
use crate::image::Image;

use crate::Float;

pub struct OlsenData {
    pub P: Vec<DMatrix<Float>>,
    pub images: Vec<Image>,
    pub U: DMatrix<Float>,
    pub point_indices: Vec<DMatrix<Float>>,
    pub image_points: Vec<DMatrix<Float>>
}

impl OlsenData {

    pub fn new(folder_path: &str) -> OlsenData {
        assert_eq!(folder_path.chars().last().unwrap(),'/');
        // Cameras - images and cameras are implicitly aligned via index
        let P = load_matrices(format!("{}{}",folder_path,"P.txt").as_str());
        let images = load_images(folder_path, "JPG");
        assert_eq!(P.len(), images.len());
        let number_of_images = images.len();
    
        // U are the reconstructed 3D points 
        let U = load_matrix(format!("{}{}",folder_path,"U.txt").as_str());
    
        // u_uncalib contains two cells; u_uncalib.points{i} contains imagepoints and u_uncalib.index{i} contains the indices of the 3D points corresponding to u_uncalib.points{i}.
        let mut u_uncalib = load_matrices(format!("{}{}",folder_path,"u_uncalib.txt").as_str());
        let point_indices = u_uncalib.split_off(number_of_images);
        let image_points = u_uncalib;
    
        assert_eq!(image_points.len(),number_of_images);
        assert_eq!(point_indices.len(),number_of_images);
    
        OlsenData{P,images,U,point_indices,image_points}
    }



}


