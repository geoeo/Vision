extern crate color_eyre;
extern crate vision;

use color_eyre::eyre::Result;
use vision::io::{olsen_loader::OlsenData};
use vision::numerics::pose;
use vision::Float;


fn main() -> Result<()> {
    color_eyre::install()?;

    println!("--------");

    let data_set_door_path = "D:/Workspace/Datasets/Olsen/Door_Lund/";
    let data_set_ahlströmer_path = "D:/Workspace/Datasets/Olsen/Jonas_Ahlströmer/";

    let olsen_data = OlsenData::new(data_set_ahlströmer_path);
    let positive_principal_distance = true;


    let (cam_intrinsics,cam_extrinsics) = olsen_data.get_camera_intrinsics_extrinsics(0,positive_principal_distance);
    let (cam_intrinsics_4,cam_extrinsics_4) = olsen_data.get_camera_intrinsics_extrinsics(4,positive_principal_distance);
    println!("{}",cam_intrinsics);
    println!("{}",cam_extrinsics);

    let pose_0 = pose::from_matrix(&cam_extrinsics);
    let pose_4 = pose::from_matrix(&cam_extrinsics_4);

    println!("{}",pose_0);
    println!("{}",pose_4);

    let pose_0_to_4 = pose::pose_difference(&pose_0, &pose_4);
    println!("{}",pose_0_to_4);
    let t = pose_0_to_4*nalgebra::Point3::<Float>::new(0.0,0.0,1.0);
    println!("{}",t);

    let (cam_intrinsics_3,cam_extrinsics_3) = olsen_data.get_camera_intrinsics_extrinsics(3,positive_principal_distance);

    // let matches_0_1 = olsen_data.get_matches_between_images(0, 10);
    // println!("matches between 0 and 1 are: #{}", matches_0_1.len());

    // let matches_0_5 = olsen_data.get_matches_between_images(0, 5);
    // println!("matches between 0 and 5 are: #{}", matches_0_5.len());
    
    // let matches_0_10 = olsen_data.get_matches_between_images(0, 10);
    // println!("matches between 0 and 10 are: #{}", matches_0_10.len());
    
    // let matches_0_20 = olsen_data.get_matches_between_images(0, 20);
    // println!("matches between 0 and 20 are: #{}", matches_0_20.len());

    Ok(())
}