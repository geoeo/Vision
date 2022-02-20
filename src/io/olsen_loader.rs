extern crate nalgebra as na;

use na::{DMatrix,Matrix4,Matrix3,Matrix3x4};
use crate::io::{ octave_loader::{load_matrices,load_matrix},load_images};
use crate::image::{Image,features::{Match,ImageFeature}};
use crate::sensors::camera::{decompose_projection,Camera,pinhole::Pinhole};

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

        let points_indices_image_one = &self.point_indices[first_index];
        let points_indices_image_two = &self.point_indices[second_index];

        let mut point_correspondence_map: Vec<(Option<usize>,Option<usize>)> = vec![(None,None);self.U.len()];

        for i in 0..points_indices_image_one.ncols() {
            let point_idx = points_indices_image_one[(0,i)];
            assert_eq!(point_idx.fract(),0.0);
            point_correspondence_map[point_idx as usize].0 = Some(i);
        }

        for i in 0..points_indices_image_two.ncols() {
            let point_idx = points_indices_image_two[(0,i)];
            assert_eq!(point_idx.fract(),0.0);
            point_correspondence_map[point_idx as usize].1 = Some(i);
        }

        point_correspondence_map.iter().filter(|&(v1,v2)| v1.is_some() && v2.is_some()).map(|&(v1,v2)| {
            let coords_one = features_img_one.column(v1.unwrap());
            let feature_one = ImageFeature::new(coords_one[0],coords_one[1]);
            let coords_two = features_img_two.column(v2.unwrap());
            let feature_two = ImageFeature::new(coords_two[0],coords_two[1]);

            Match{feature_one,feature_two}
        }).collect::<Vec<Match<ImageFeature>>>()
    }

    pub fn get_camera_intrinsics_extrinsics(&self, image_index: usize,  positive_principal_distance: bool) -> (Matrix3<Float>, Matrix4<Float>){
        let projection = &self.P[image_index];
        assert_eq!(projection.nrows(),3);
        assert_eq!(projection.ncols(),4);
        let mut projection_static = Matrix3x4::<Float>::zeros();  
        projection_static.fixed_columns_mut::<4>(0).copy_from(&projection.fixed_columns::<4>(0));

        decompose_projection(&projection_static,positive_principal_distance)
    }

    pub fn get_image_dim(&self) -> (usize,usize) {
        assert!(self.images.len() > 0);
        let (r,c) = self.images[0].get_rows_columns();
        for im in &self.images {
            assert_eq!((r,c), im.get_rows_columns());
        }
        (r,c)

    }



}




