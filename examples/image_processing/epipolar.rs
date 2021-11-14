extern crate image as image_rs;
extern crate vision;
extern crate color_eyre;

use std::fs;
use color_eyre::eyre::Result;
use std::path::Path;
use vision::image::pyramid::orb::{orb_runtime_parameters::OrbRuntimeParameters};
use vision::image::features::{Match,orb_feature::OrbFeature};
use vision::image::Image;
use vision::image::bundle_adjustment::{camera_feature_map::CameraFeatureMap, solver::optimize};
use vision::image::epipolar;
use vision::sensors::camera::{pinhole::Pinhole, Camera};
use vision::odometry::runtime_parameters::RuntimeParameters;
use vision::numerics::{loss, weighting};


fn main() -> Result<()> {

    color_eyre::install()?;

    let image_name_1 = "ba_slow_1";
    let image_name_2 = "ba_slow_2";

    let image_name_3 = "ba_slow_3";
    let image_name_4 = "ba_slow_4";


    let image_format = "png";
    let image_folder = "images/";
    let image_out_folder = "output/";
    let image_path_1 = format!("{}{}.{}",image_folder,image_name_1, image_format);
    let image_path_2 = format!("{}{}.{}",image_folder,image_name_2, image_format);
    let image_path_3 = format!("{}{}.{}",image_folder,image_name_3, image_format);
    let image_path_4 = format!("{}{}.{}",image_folder,image_name_4, image_format);

    let gray_image_1 = image_rs::open(&Path::new(&image_path_1)).unwrap().to_luma8();
    let gray_image_2 = image_rs::open(&Path::new(&image_path_2)).unwrap().to_luma8();
    let gray_image_3 = image_rs::open(&Path::new(&image_path_3)).unwrap().to_luma8();
    let gray_image_4 = image_rs::open(&Path::new(&image_path_4)).unwrap().to_luma8();

    let image_1 = Image::from_gray_image(&gray_image_1, false, false, Some(image_name_1.to_string()));
    let image_2 = Image::from_gray_image(&gray_image_2, false, false, Some(image_name_2.to_string()));
    let image_3 = Image::from_gray_image(&gray_image_3, false, false, Some(image_name_3.to_string()));
    let image_4 = Image::from_gray_image(&gray_image_4, false, false, Some(image_name_4.to_string()));


    //TODO: camera intrinsics -investigate removing badly matched feature in the 2 image set
    let intensity_camera_1 = Pinhole::new(389.2685546875, 389.2685546875, 319.049255371094, 241.347015380859, true);
    let intensity_camera_2 = intensity_camera_1.clone();
    let cameras = vec!(intensity_camera_1,intensity_camera_2);


    let orb_matches_read = fs::read_to_string("D:/Workspace/Rust/Vision/output/orb_ba_matches_2_images.txt").expect("Unable to read file");
    let matches: Vec<Vec<Match<OrbFeature>>> = serde_yaml::from_str(&orb_matches_read)?;

    let fundamental_matrix = epipolar::eight_point(&matches[0]);
    let essential_matrix = epipolar::compute_essential(&fundamental_matrix, &intensity_camera_1.get_projection(), &intensity_camera_2.get_projection());
    let (h,R) = epipolar::decompose_essential(&essential_matrix, &matches[0]);

   



    Ok(())


}