extern crate nalgebra as na;
extern crate vision;

use na::UnitQuaternion;
use vision::io::{loading_parameters::LoadingParameters,d455_loader};

fn main() {


    let dataset_name = "simple_trans";


    let root_path = format!("C:/Users/Marc/Workspace/Datasets/D455/{}",dataset_name);
    let out_folder = "C:/Users/Marc/Workspace/Rust/Vision/output";


    let loading_parameters = LoadingParameters {
        starting_index: 0,
        step :1,
        count :20,
        negate_values :true,
        invert_focal_lengths :true,
        invert_y :true,
        gt_alignment_rot:UnitQuaternion::identity()
    };

    let loaded_data = d455_loader::load(&root_path, &loading_parameters);





}