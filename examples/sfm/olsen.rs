extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;
use vision::io::{olsen_loader::OlsenData};


fn main() -> Result<()> {
    color_eyre::install()?;


    let olsen_data = OlsenData::new("D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/");

    for im in olsen_data.images {
        println!("{}", im.name.unwrap());
    }


    Ok(())
}