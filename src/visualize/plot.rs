extern crate plotters;
extern crate nalgebra as na;

use plotters::prelude::*;
use na::Vector3;

use crate::{float,Float};

fn get_min_max(data_vectors: Vec<&Vec<Float>>) -> (Float,Float) {

    let mut min = float::MAX;
    let mut max = float::MIN;
    
    for i in 0..data_vectors.len()-1 {
        assert_eq!(data_vectors[i].len(),data_vectors[i+1].len());
    }

    for j in 0..data_vectors.len() {
        let data = data_vectors[j];
        for i in 0..data.len() {
            let v = data[i];

            if v < min {
                min = v;
            }
    
            if v > max {
                max = v;
            }
        }

    }

    if(max-min) < 1e-5 {
        max = min + 1e-5;
    }

    (min,max)
}

pub fn draw_line_graph(data: &Vec<Float>, output_folder: &str, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (min,max) = get_min_max(vec!(data));

    let path = format!("{}/{}",output_folder,file_name);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;


    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption("Area Chart Demo", ("sans-serif", 40))
        .build_cartesian_2d(0..(data.len() - 1), min..max)?; 

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

pub fn draw_line_graph_est_gt(data_est: &Vec<Float>,data_gt: &Vec<Float>, output_folder: &str, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (min,max) = get_min_max(vec!(data_est,data_gt));


    let path = format!("{}/{}",output_folder,file_name);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption("Translation", ("sans-serif", 40))
        .build_cartesian_2d(0..(data_est.len() - 1), min..max)?; 

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    chart.draw_series(
        LineSeries::new(
            (0..).zip(data_est.iter()).map(|(x, y)| (x, *y)),
            &RED.mix(0.2),
        )
    )?;

    chart.draw_series(
        LineSeries::new(
            (0..).zip(data_gt.iter()).map(|(x, y)| (x, *y)),
            &GREEN.mix(0.2),
        )
    )?;

    Ok(())
}

pub fn draw_line_graph_translation_est_gt(translation_est: &Vec<Vector3<Float>>,translation_gt: &Vec<Vector3<Float>>, output_folder: &str, file_name: &str, info: &str) -> Result<(), Box<dyn std::error::Error>> {
    let x_translation_est = translation_est.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let x_translation_gt = translation_gt.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let y_translation_est = translation_est.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let y_translation_gt = translation_gt.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let z_translation_est = translation_est.iter().map(|point| point[2]).collect::<Vec<Float>>();
    let z_translation_gt = translation_gt.iter().map(|point| point[2]).collect::<Vec<Float>>();

    let data_est_translation = vec!(x_translation_est,y_translation_est,z_translation_est);
    let data_gt_translation = vec!(x_translation_gt,y_translation_gt,z_translation_gt);

    let path = format!("{}/{}",output_folder,file_name);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    root.titled(info, ("sans-serif", 12))?;


    let drawing_areas = root.split_evenly((3,1));


    for i in 0..drawing_areas.len() {

        let (min,max) = get_min_max(vec!(&data_est_translation[i],&data_gt_translation[i]));

        let title = match i {
            0 => "Translation X",
            1 => "Translation Y", 
            2 => "Translation Z",
            _ => panic!("unexpected plot index")
        };

        let mut chart = ChartBuilder::on(&drawing_areas[i])
        .margin(30)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(0..(data_est_translation[i].len() - 1), min..max)?; 

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_est_translation[i].iter()).map(|(x, y)| (x, *y)),
                &RED.mix(0.2),
            )
        )?;

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_gt_translation[i].iter()).map(|(x, y)| (x, *y)),
                &GREEN.mix(0.2),
            )
        )?;

    }



    Ok(())
}

pub fn draw_line_graph_vector3(translation_est: &Vec<Vector3<Float>>, output_folder: &str, file_name: &str, title: &str, subtitle_header: &str, y_desc: &str) -> Result<(), Box<dyn std::error::Error>> {
    let x_translation_est = translation_est.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let y_translation_est = translation_est.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let z_translation_est = translation_est.iter().map(|point| point[2]).collect::<Vec<Float>>();
    let data_est_translation = vec!(x_translation_est,y_translation_est,z_translation_est);


    let path = format!("{}/{}",output_folder,file_name);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    root.titled(title, ("sans-serif", 12))?;

    let drawing_areas = root.split_evenly((3,1));


    for i in 0..drawing_areas.len() {

        let (min,max) = get_min_max(vec!(&data_est_translation[i]));

        let title = match i {
            0 => format!("{} X", subtitle_header),
            1 => format!("{} Y", subtitle_header), 
            2 => format!("{} Z", subtitle_header),
            _ => panic!("unexpected plot index")
        };

        let mut chart = ChartBuilder::on(&drawing_areas[i])
        .margin(30)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(0..(data_est_translation[i].len() - 1), min..max)?; 

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .y_desc(y_desc)
            .draw()?;

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_est_translation[i].iter()).map(|(x, y)| (x, *y)),
                &RED.mix(0.2),
            )
        )?;

    }
    Ok(())
}

pub fn draw_line_graph_two_vector3(translation_est: &Vec<Vector3<Float>>,label_1: &str, translation_est_2: &Vec<Vector3<Float>>,label_2: &str, output_folder: &str, file_name: &str, title: &str, subtitle_header: &str, y_desc: &str) -> Result<(), Box<dyn std::error::Error>> {
    let x_translation_est = translation_est.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let y_translation_est = translation_est.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let z_translation_est = translation_est.iter().map(|point| point[2]).collect::<Vec<Float>>();
    let data_est_translation = vec!(x_translation_est,y_translation_est,z_translation_est);

    let x_translation_est_2 = translation_est_2.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let y_translation_est_2 = translation_est_2.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let z_translation_est_2 = translation_est_2.iter().map(|point| point[2]).collect::<Vec<Float>>();
    let data_est_translation_2 = vec!(x_translation_est_2,y_translation_est_2,z_translation_est_2);


    let path = format!("{}/{}",output_folder,file_name);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    root.titled(title, ("sans-serif", 12))?;

    let drawing_areas = root.split_evenly((3,1));


    for i in 0..drawing_areas.len() {

        let (min_1,max_1) = get_min_max(vec!(&data_est_translation[i]));
        let (min_2,max_2) = get_min_max(vec!(&data_est_translation_2[i]));
        

        let min = match (min_1,min_2) {
            (min_1,min_2) if min_1 > min_2 => min_2,
            _ => min_1
        };
        let max = match (max_1,max_2) {
            (max_1,max_2) if max_1 > max_2 => max_1,
            _ => max_2
        };

        let title = match i {
            0 => format!("{} X", subtitle_header),
            1 => format!("{} Y", subtitle_header), 
            2 => format!("{} Z", subtitle_header),
            _ => panic!("unexpected plot index")
        };

        let mut chart = ChartBuilder::on(&drawing_areas[i])
        .margin(30)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(0..(data_est_translation[i].len() - 1), min..max)?; 

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .y_desc(y_desc)
            .draw()?;

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_est_translation[i].iter()).map(|(x, y)| (x, *y)),
                &RED.mix(0.2),
            )
        )?.label(label_1).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_est_translation_2[i].iter()).map(|(x, y)| (x, *y)),
                &BLUE.mix(0.2),
            )
        )?.label(label_2).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.configure_series_labels().border_style(&BLACK).draw()?;


    }
    Ok(())
}

pub fn draw_line_graph_three_vector3(translation_est: &Vec<Vector3<Float>>,label_1: &str, translation_est_2: &Vec<Vector3<Float>>,label_2: &str,translation_est_3: &Vec<Vector3<Float>>, label_3: &str, output_folder: &str, file_name: &str, title: &str, subtitle_header: &str, y_desc: &str) -> Result<(), Box<dyn std::error::Error>> {
    let x_translation_est = translation_est.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let y_translation_est = translation_est.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let z_translation_est = translation_est.iter().map(|point| point[2]).collect::<Vec<Float>>();
    let data_est_translation = vec!(x_translation_est,y_translation_est,z_translation_est);

    let x_translation_est_2 = translation_est_2.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let y_translation_est_2 = translation_est_2.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let z_translation_est_2 = translation_est_2.iter().map(|point| point[2]).collect::<Vec<Float>>();
    let data_est_translation_2 = vec!(x_translation_est_2,y_translation_est_2,z_translation_est_2);

    
    let x_translation_est_3 = translation_est_3.iter().map(|point| point[0]).collect::<Vec<Float>>();
    let y_translation_est_3 = translation_est_3.iter().map(|point| point[1]).collect::<Vec<Float>>();
    let z_translation_est_3 = translation_est_3.iter().map(|point| point[2]).collect::<Vec<Float>>();
    let data_est_translation_3 = vec!(x_translation_est_3,y_translation_est_3,z_translation_est_3);


    let path = format!("{}/{}",output_folder,file_name);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    root.titled(title, ("sans-serif", 12))?;

    let drawing_areas = root.split_evenly((3,1));


    for i in 0..drawing_areas.len() {

        let (min_1,max_1) = get_min_max(vec!(&data_est_translation[i]));
        let (min_2,max_2) = get_min_max(vec!(&data_est_translation_2[i]));
        let (min_3,max_3) = get_min_max(vec!(&data_est_translation_3[i]));
        

        let min = match (min_1,min_2, min_3) {
            (min_1,min_2, min_3) if min_1 > min_2 && min_3 > min_2 => min_2,
            (min_1,min_2, min_3) if min_1 > min_3 && min_2 > min_3 => min_3,
            _ => min_1
        };
        let max = match (max_1,max_2, max_3) {
            (max_1,max_2, max_3) if max_1 > max_2 && max_1 > max_3 => max_1,
            (max_1,max_2, max_3) if max_2 > max_3 && max_2 > max_1 => max_2,
            _ => max_3
        };

        let title = match i {
            0 => format!("{} X", subtitle_header),
            1 => format!("{} Y", subtitle_header), 
            2 => format!("{} Z", subtitle_header),
            _ => panic!("unexpected plot index")
        };

        let mut chart = ChartBuilder::on(&drawing_areas[i])
        .margin(30)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(0..(data_est_translation[i].len() - 1), min..max)?; 

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .y_desc(y_desc)
            .draw()?;

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_est_translation[i].iter()).map(|(x, y)| (x, *y)),
                &RED.mix(0.2),
            )
        )?.label(label_1).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_est_translation_2[i].iter()).map(|(x, y)| (x, *y)),
                &BLUE.mix(0.2),
            )
        )?.label(label_2).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.draw_series(
            LineSeries::new(
                (0..).zip(data_est_translation_3[i].iter()).map(|(x, y)| (x, *y)),
                &CYAN.mix(0.2),
            )
        )?.label(label_3).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));

        chart.configure_series_labels().border_style(&BLACK).draw()?;


    }
    Ok(())
}