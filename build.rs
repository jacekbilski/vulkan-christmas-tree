extern crate shaderc;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Tell the build script to only run again if we change our source shaders
    println!("cargo:rerun-if-changed=src/shaders");

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));

    // Create destination path if necessary
    std::fs::create_dir_all("target/shaders")?;

    for entry in std::fs::read_dir("src/shaders")? {
        let entry = entry?;

        if entry.file_type()?.is_file() {
            let in_path = entry.path();

            // Support only vertex and fragment shaders currently
            let shader_type =
                in_path
                    .extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "vert" => Some(shaderc::ShaderKind::Vertex),
                        "frag" => Some(shaderc::ShaderKind::Fragment),
                        "comp" => Some(shaderc::ShaderKind::Compute),
                        _ => None,
                    });
            if let Some(shader_type) = shader_type {
                let source = std::fs::read_to_string(&in_path)?;
                let binary_result = compiler.compile_into_spirv(
                    &source,
                    shader_type,
                    in_path.file_name().unwrap().to_str().unwrap(),
                    "main",
                    Some(&options),
                )?;
                let out_path = format!(
                    "target/shaders/{}.spv",
                    in_path.file_name().unwrap().to_string_lossy()
                );
                std::fs::write(&out_path, binary_result.as_binary_u8())?;
            }
        }
    }

    Ok(())
}
