extern crate nalgebra as na;

use na::DMatrix;
use crate::io::{ octave_loader::{load_matrices,load_matrix},load_images};
use crate::image::{Image,features::{Match,ImageFeature}};

use crate::Float;

#[allow(non_snake_case)]
pub struct OlsenData {
    pub P: Vec<DMatrix<Float>>,
    pub images: Vec<Image>,
    pub U: DMatrix<Float>,
    pub point_indices: Vec<DMatrix<Float>>,
    pub image_points: Vec<DMatrix<Float>>
}

impl OlsenData {

    #[allow(non_snake_case)]
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

    pub fn get_matches_between_images(&self, first_index: usize, second_index: usize) -> Vec<Match<ImageFeature>> {

        let features_img_one = &self.image_points[first_index];
        let features_img_two = &self.image_points[second_index];

        let points_indices_image_one = &self.image_points[first_index];
        let points_indices_image_two = &self.image_points[second_index];

        let mut tmp_map = vec![(false,false);self.U.len()];

        for i in 0..points_indices_image_one.ncols() {
            let point_idx = points_indices_image_one[(0,i)];
            assert_eq!(point_idx.fract(),0.0);
            tmp_map[point_idx as usize].0 = true
        }

        for i in 0..points_indices_image_two.ncols() {
            let point_idx = points_indices_image_two[(0,i)];
            assert_eq!(point_idx.fract(),0.0);
            tmp_map[point_idx as usize].1 = true
        }

        //TODO: check this
        tmp_map.iter().enumerate().filter(|(_,&(a,b))| a && b).map(|(i,_)| {
            let coords_one = features_img_one.column(i);
            let feature_one = ImageFeature::new(coords_one[0],coords_one[1]);
            let coords_two = features_img_two.column(i);
            let feature_two = ImageFeature::new(coords_two[0],coords_two[1]);

            Match{feature_one,feature_two}
        }).collect::<Vec<Match<ImageFeature>>>()
    }


}




