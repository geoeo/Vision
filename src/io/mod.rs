pub mod eth_loader;
pub mod tum_loader;
pub mod loading_parameters;
pub mod loaded_data;

use crate::Float;

pub fn parse_to_float(string: &str, negate_value: bool) -> Float {
    let parts = string.split("e").collect::<Vec<&str>>();
    let factor = match negate_value {
        true => -1.0,
        false => 1.0
    };
    match parts.len() {
        1 => factor * parts[0].parse::<Float>().unwrap(),
        2 => {
            let num = parts[0].parse::<Float>().unwrap();
            let exponent = parts[1].parse::<i32>().unwrap();
            factor * num*(10f64.powi(exponent) as Float)
        },
        _ => panic!("string malformed for parsing to float: {}", string)
    }
}