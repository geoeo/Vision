extern crate plotters;
use plotters::prelude::*;

use crate::Float;

pub fn draw_line_graph(data: &Vec<Float>, output_folder: &str, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {


    let path = format!("{}/{}",output_folder,file_name);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption("Area Chart Demo", ("sans-serif", 40))
        .build_cartesian_2d(0..(data.len() - 1), 0.0..1500.0)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    chart.draw_series(
        LineSeries::new(
            (0..).zip(data.iter()).map(|(x, y)| (x, *y)),
            &RED.mix(0.2),
        )
    )?;

    Ok(())


}