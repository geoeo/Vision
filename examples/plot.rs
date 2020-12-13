use rand_distr::{Distribution, Normal};

use vision::visualize::plot::draw_line_graph;

pub fn main() {

    let data: Vec<_> = {
        let norm_dist = Normal::new(500.0, 100.0).unwrap();
        let mut sampling_thread = rand::thread_rng();
        let x_iter = norm_dist.sample_iter(&mut sampling_thread);
        x_iter
            .filter(|x| *x < 1500.0)
            .take(100)
            .zip(0..)
            .map(|(x, b)| x + (b as f64).powf(1.2))
            .collect()
    };

    draw_line_graph(&data, "output", "area-chart.png");
}