extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;
use vision::io::{octave_loader::load_matrices, olsen_loader::load_images};


fn main() -> Result<()> {
    color_eyre::install()?;


    let data = load_matrices("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/u_uncalib.txt");
    let images = load_images("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/ahlstromer/");

    for im in images {
        println!("{}", im.name.unwrap());
    }


    Ok(())
}