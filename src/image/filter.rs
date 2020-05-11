use std::f64::consts::PI;

fn gaussian_sample(mean: f64, std: f64, x:f64) -> f64 {
    let exponent = ((x-mean)/(-2.0*std)).powi(2).exp();
    let factor = 1.0/(std*(2.0*PI).sqrt());
    factor*exponent
}

fn gaussian_1_d_kernel(mean: f64, std: f64, step: i8, end: i8) -> Vec<f64> {
    assert_eq!(end%step,0);
    assert!(end > 0);

    let range = (-end..end+1).step_by(step as usize);
    range.map(|x| gaussian_sample(mean,std,x as f64)).collect()
}

