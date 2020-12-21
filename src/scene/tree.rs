use std::f32::consts::PI;

use cgmath::{vec3, Matrix4, Point3, Rad};
use tobj::{load_mtl_buf, load_obj_buf};

use crate::mesh::{Color, InstanceData, Mesh};
use crate::vulkan::Vertex;

pub fn create_meshes() -> Vec<Mesh> {
    let object_source = include_str!("../../models/tree.obj");
    let materials_source = include_str!("../../models/tree.mtl");
    let tree = load_obj_buf(&mut object_source.as_bytes(), false, |_| {
        load_mtl_buf(&mut materials_source.as_bytes())
    });
    let (models, model_materials) = tree.unwrap();
    let mut meshes: Vec<Mesh> = vec![];
    for mi in 0..models.len() {
        let mut vertices: Vec<Vertex> = vec![];
        let mut indices: Vec<u32> = vec![];
        let mesh = models[mi].mesh.clone();
        for vi in (0..mesh.positions.len()).step_by(3) {
            let position = Point3::new(
                mesh.positions[vi],
                mesh.positions[vi + 1],
                mesh.positions[vi + 2],
            );
            let normal = vec3(mesh.normals[vi], mesh.normals[vi + 1], mesh.normals[vi + 2]);
            vertices.push(Vertex {
                pos: position.into(),
                norm: normal.into(),
            });
        }
        indices.extend(mesh.indices.iter());
        let material = &model_materials[models[mi].mesh.material_id.unwrap()];
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
