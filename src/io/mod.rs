pub mod tum_loader;

use crate::Float;

pub fn parse_to_float(string: &str) -> Float {
    let parts = string.split("e").collect::<Vec<&str>>();
    match parts.len() {
        1 => parts[0].parse::<Float>().unwrap(),
        2 => {
            let num = parts[0].parse::<Float>().unwrap();
            let exponent = parts[1].parse::<i32>().unwrap();
            num*(10f64.powi(exponent) as Float)
        },
        _ => panic!("string malformed for parsing to float: {}", string)
    }
}