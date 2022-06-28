use std::f32::consts::PI;

use cgmath::{vec3, Matrix4, Point3, Rad};
use tobj::{load_mtl_buf, load_obj_buf};

use crate::mesh::{Color, InstanceData, Mesh};
use crate::vulkan::Vertex;

pub fn create_meshes() -> Vec<Mesh> {
    let object_source = include_str!("../../models/tree.obj");
    let materials_source = include_str!("../../models/tree.mtl");
    let load_options = tobj::LoadOptions {
        triangulate: true,
        ..Default::default()
    };
    let tree = load_obj_buf(&mut object_source.as_bytes(), &load_options, |_| {
        load_mtl_buf(&mut materials_source.as_bytes())
    });
    let (models, model_materials) = tree.unwrap();
    let materials = model_materials.unwrap();
    let mut meshes: Vec<Mesh> = vec![];
    for mi in 0..models.len() {
        let mut vertices: Vec<Vertex> = vec![];
        let mut indices: Vec<u32> = vec![];
        let mesh = models[mi].mesh.clone();

        for i in 0..mesh.indices.len() {
            let pi = 3 * mesh.indices[i] as usize;
            let position = Point3::new(
                mesh.positions[pi],
                mesh.positions[pi + 1],
                mesh.positions[pi + 2],
            );
            let ni = 3 * mesh.normal_indices[i] as usize;
            let normal = vec3(mesh.normals[ni], mesh.normals[ni + 1], mesh.normals[ni + 2]);

            vertices.push(Vertex {
                pos: position.into(),
                norm: normal.into(),
            });
        }
        indices.extend((0..mesh.indices.len() as u32).into_iter());
        let material = &materials[models[mi].mesh.material_id.unwrap()];
        let color = Color {
            ambient: material.ambient,
            diffuse: material.diffuse,
            specular: material.specular,
            shininess: material.shininess,
        };
        let model: Matrix4<f32> =
            Matrix4::from_angle_z(Rad(PI)) * Matrix4::from_nonuniform_scale(1.8, 1., 1.8);
        let instance = InstanceData {
            color,
            model,
            ..Default::default()
        };
        let mesh = Mesh {
            vertices,
            indices,
            instances: vec![instance],
        };
        meshes.push(mesh);
    }
    meshes
}
