extern crate nalgebra as na;
extern crate image as image_rs;

use na::{Vector3, Quaternion};
use crate::image::Image;
use crate::sensors::camera::perspective::Perspective;
use crate::Float;


#[derive(Clone)]
pub struct CameraDataFrame {
    pub source_timestamps: Vec<Float>,
    pub target_timestamps: Vec<Float>,
    pub source_gray_images: Vec<Image>,
    pub source_depth_images: Vec<Image>,
    pub target_gray_images: Vec<Image>,
    pub target_depth_images: Vec<Image>,
    pub intensity_camera: Perspective<Float>,
    pub depth_camera: Perspective<Float>,
    pub target_gt_poses: Option<Vec<(Vector3<Float>,Quaternion<Float>)>>,
    pub source_gt_poses: Option<Vec<(Vector3<Float>,Quaternion<Float>)>> 
}