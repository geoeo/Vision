
use std::path::Path;
use std::fs::File;
use std::io::{BufReader,BufRead};
use crate::image::features::{matches::Match,image_feature::ImageFeature};
use crate::io;


pub fn load_matches(root_path: &str, file_name_1: &str, file_name_2: &str) -> Vec<Match<ImageFeature>> {
    let file_1_path_str = format!("{}/{}",root_path,file_name_1);
    let file_2_path_str = format!("{}/{}",root_path,file_name_2);
    let file_1_path = Path::new(&file_1_path_str);
    let file_2_path = Path::new(&file_2_path_str);

    let file_1 = File::open(file_1_path).expect(format!("Could not open: {}", file_1_path.display()).as_str());
    let file_2 = File::open(file_2_path).expect(format!("Could not open: {}", file_2_path.display()).as_str());


    let reader = BufReader::new(file_1);
    let lines = reader.lines();
    let features_1 = lines.map(|l| {
        let v = l.unwrap();
        let values = v.trim().split(' ').collect::<Vec<&str>>();
        let x = io::parse_to_float(values[0], false); 
        let y = io::parse_to_float(values[1], false); 
        ImageFeature::new(x,y, None)
    }).collect::<Vec<ImageFeature>>();

    let reader_2 = BufReader::new(file_2);
    let lines_2 = reader_2.lines();
    let features_2 = lines_2.map(|l| {
        let v = l.unwrap();
        let values = v.trim().split(' ').collect::<Vec<&str>>();
        let x = io::parse_to_float(values[0], false); 
        let y = io::parse_to_float(values[1], false); 
        ImageFeature::new(x,y, None)
    }).collect::<Vec<ImageFeature>>();

    features_1.into_iter().zip(features_2.into_iter()).map(|(f1,f2)| {
        Match::new(f1, f2)
    }).collect::<Vec<Match<ImageFeature>>>()
}