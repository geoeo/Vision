extern crate image as image_rs;

use image_rs::GrayImage;

use crate::image::Image;
use crate::pyramid::Pyramid;
use self::{orb_octave::OrbOctave, orb_runtime_parameters::OrbRuntimeParameters};
use crate::features::orb_feature::OrbFeature;
use crate::matching::brief_descriptor::BriefDescriptor;


pub mod orb_octave;
pub mod orb_runtime_parameters;

pub fn build_orb_pyramid(base_gray_image: &GrayImage, runtime_parameters: &OrbRuntimeParameters) -> Pyramid<OrbOctave> {

    let mut octaves: Vec<OrbOctave> = Vec::with_capacity(runtime_parameters.octave_count);
    let base_image = Image::from_gray_image(base_gray_image, false);

    let mut octave_image = base_image;
    let mut sigma = runtime_parameters.sigma;

    for i in 0..runtime_parameters.octave_count {

        if i > 0 {
            octave_image = Image::downsample_half(&octaves[i-1].images[0], false,  runtime_parameters.min_image_dimensions);
            sigma *= 2.0;
        }

        let new_octave = OrbOctave::build_octave(&octave_image,sigma, runtime_parameters);

        octaves.push(new_octave);
    }

    Pyramid {octaves}
}

pub fn generate_features_for_octave(octave: &OrbOctave, runtime_parameters: &OrbRuntimeParameters) -> Vec<OrbFeature> {
    OrbFeature::new(&octave.images, runtime_parameters.fast_circle_radius, runtime_parameters.fast_threshold_factor, runtime_parameters.fast_consecutive_pixels, runtime_parameters.fast_grid_size, runtime_parameters.harris_k)
}

pub fn generate_feature_pyramid(pyramid: &Pyramid<OrbOctave>, runtime_parameters: &OrbRuntimeParameters) -> Pyramid<Vec<OrbFeature>> {
    Pyramid{octaves: pyramid.octaves.iter().map(|x| generate_features_for_octave(x,runtime_parameters)).collect::<Vec<Vec<OrbFeature>>>()}
}

pub fn generate_feature_descriptor_pyramid(octave_pyramid: &Pyramid<OrbOctave>, feature_pyramid: &Pyramid<Vec<OrbFeature>>, runtime_parameters: &OrbRuntimeParameters) -> Pyramid<Vec<(OrbFeature,BriefDescriptor)>> {
    assert_eq!(octave_pyramid.octaves.len(),feature_pyramid.octaves.len());
    let octave_len = octave_pyramid.octaves.len();
    let mut feature_descriptor_pyramid = Pyramid::<Vec<(OrbFeature,BriefDescriptor)>>::empty(octave_len);

    for i in 0..octave_len {
        let image = &octave_pyramid.octaves[i].images[0];
        let feature_octave = &feature_pyramid.octaves[i];
        let data_vector 
            = feature_octave.iter()
                            .enumerate()
                            .map(|x| (x.0,BriefDescriptor::new(image, x.1, runtime_parameters.brief_n, runtime_parameters.brief_s)))
                            .filter(|x| x.1.is_some())
                            .map(|(idx,option)| (feature_octave[idx],option.unwrap().0))
                            .collect::<Vec<(OrbFeature,BriefDescriptor)>>();

        if data_vector.len() == 0 {
            println!("Warning: 0 features with descriptors for octave idx: {}",i);
        }

        feature_descriptor_pyramid.octaves.push(data_vector);
    }

    feature_descriptor_pyramid
}

pub fn generate_match_pyramid(feature_descriptor_pyramid_a: &Pyramid<Vec<(OrbFeature,BriefDescriptor)>>,feature_descriptor_pyramid_b: &Pyramid<Vec<(OrbFeature,BriefDescriptor)>>) -> Pyramid<Vec<(OrbFeature,OrbFeature)>> {
    let octave_len = feature_descriptor_pyramid_a.octaves.len();
    let mut match_pyramid = Pyramid::<Vec<(OrbFeature,OrbFeature)>>::empty(octave_len);

    for i in 0..octave_len {
        let descriptors_a = feature_descriptor_pyramid_a.octaves[i].iter().map(|(_,descriptor)| descriptor).collect::<Vec<&BriefDescriptor>>();
        let descriptors_b = feature_descriptor_pyramid_b.octaves[i].iter().map(|(_,descriptor)| descriptor).collect::<Vec<&BriefDescriptor>>();
        let matches_for_a_in_b = BriefDescriptor::match_descriptors(&descriptors_a, &descriptors_b);

        let feature_descriptors_a = feature_descriptor_pyramid_a.octaves[i].iter().map(|x| x.0).collect::<Vec<OrbFeature>>();
        let feature_descriptors_b = &feature_descriptor_pyramid_b.octaves[i];
        let features_b_aligned_to_a = matches_for_a_in_b.iter().map(|&x| feature_descriptors_b[x].0).collect::<Vec<OrbFeature>>();
        
        let matches = feature_descriptors_a.into_iter().zip(features_b_aligned_to_a.into_iter()).collect::<Vec<(OrbFeature,OrbFeature)>>();
        match_pyramid.octaves.push(matches);
    }

    match_pyramid

}



