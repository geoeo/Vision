use color_eyre::eyre::Result;
use models_cv::io::deserialize_feature_matches;

fn main() -> Result<()> {
    color_eyre::install()?;
    let runtime_conf = load_runtime_conf();
    let file_name = "feature_matches_Suzanne.yaml";
    let path = format!("{}/{}",runtime_conf.local_data_path,file_name);
    let loaded_data = models_cv::io::deserialize_feature_matches(&path);
    println!("{:?}",loaded_data);

}