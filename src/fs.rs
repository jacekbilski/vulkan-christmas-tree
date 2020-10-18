use std::path::Path;

/// Contains helper functions related to filesystem operations

pub fn read_shader_code(file_name: &str) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;

    let path_string = "target/shaders/".to_owned() + file_name;
    let spv_file = File::open(Path::new(&path_string))
        .expect(&format!("Failed to find spv file at {:?}", path_string));
    let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

    bytes_code
}
