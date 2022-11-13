use cgmath::{Matrix4, SquareMatrix};

use crate::textured_mesh::{InstanceData, TexturedMesh, TexturedVertex};
use crate::vulkan::VertexIndexType;

const VERTICES_DATA: [TexturedVertex; 4] = [
    TexturedVertex {
        pos: [-10., 5., -10.],
        norm: [0., -1., 0.],
    },
    TexturedVertex {
        pos: [-10., 5., 10.],
        norm: [0., -1., 0.],
    },
    TexturedVertex {
        pos: [10., 5., -10.],
        norm: [0., -1., 0.],
    },
    TexturedVertex {
        pos: [10., 5., 10.],
        norm: [0., -1., 0.],
    },
];
const INDICES_DATA: [VertexIndexType; 6] = [0, 2, 1, 1, 2, 3];

pub fn create_meshes() -> Vec<TexturedMesh> {
    vec![TexturedMesh {
        vertices: Vec::from(VERTICES_DATA),
        indices: Vec::from(INDICES_DATA),
        instances: vec![InstanceData {
            model: Matrix4::identity(),
            ..Default::default()
        }],
    }]
}
