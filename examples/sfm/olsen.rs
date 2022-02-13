extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;
use vision::io::{olsen_loader::OlsenData};


fn main() -> Result<()> {
    color_eyre::install()?;


    let olsen_data = OlsenData::new("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/");

    let matches_0_1 = olsen_data.get_matches_between_images(0, 1);
    println!("matches between 0 and 1 are: #{}", matches_0_1.len());
    let cam0 = olsen_data.get_camera(0);

    let matches_0_5 = olsen_data.get_matches_between_images(0, 5);
    println!("matches between 0 and 5 are: #{}", matches_0_5.len());
    
    let matches_0_10 = olsen_data.get_matches_between_images(0, 10);
    println!("matches between 0 and 10 are: #{}", matches_0_10.len());
    
    let matches_0_20 = olsen_data.get_matches_between_images(0, 20);
    println!("matches between 0 and 20 are: #{}", matches_0_20.len());

    Ok(())
}