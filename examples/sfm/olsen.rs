extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;
use vision::io::{octave_loader::{load_matrices,load_matrix}, load_images};


fn main() -> Result<()> {
    color_eyre::install()?;



    // Cameras - images and cameras are implicitly aligned via index
    let P = load_matrices("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/P.txt");
    let images = load_images("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/ahlstromer/", "JPG");
    assert_eq!(P.len(), images.len());
    let number_of_images = images.len();

    // U are the reconstructed 3D points 
    let U = load_matrix("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/U.txt");

    // u_uncalib contains two cells; u_uncalib.points{i} contains imagepoints and u_uncalib.index{i} contains the indices of the 3D points corresponding to u_uncalib.points{i}.
    let mut u_uncalib = load_matrices("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/u_uncalib.txt");
    let point_indices = u_uncalib.split_off(number_of_images);
    let image_points = u_uncalib;

    assert_eq!(image_points.len(),number_of_images);
    assert_eq!(point_indices.len(),number_of_images);


    for im in images {
        println!("{}", im.name.unwrap());
    }


    Ok(())
}