use cgmath::{Matrix4, SquareMatrix};

use crate::mesh::{Color, InstanceData, Mesh};
use crate::vulkan::{Vertex, VertexIndexType};

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-10., 5., -10.],
        norm: [0., -1., 0.],
    },
    Vertex {
        pos: [-10., 5., 10.],
        norm: [0., -1., 0.],
    },
    Vertex {
        pos: [10., 5., -10.],
        norm: [0., -1., 0.],
    },
    Vertex {
        pos: [10., 5., 10.],
        norm: [0., -1., 0.],
    },
];
const INDICES_DATA: [VertexIndexType; 6] = [0, 2, 1, 1, 2, 3];

pub fn create_mesh() -> Mesh {
    let color = Color {
        ambient: [1.0, 1.0, 1.0],
        diffuse: [0.623960, 0.686685, 0.693872],
        specular: [0.5, 0.5, 0.5],
        shininess: 225.0,
    };
    Mesh {
        vertices: Vec::from(VERTICES_DATA),
        indices: Vec::from(INDICES_DATA),
        instances: vec![InstanceData {
            color,
            model: Matrix4::identity(),
        }],
    }
}
