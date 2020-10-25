use cgmath::{Matrix4, SquareMatrix};

use crate::mesh::{Color, InstanceData, Mesh};
use crate::vulkan::{Vertex, VertexIndexType};

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-10., 5., -10.],
    },
    Vertex {
        pos: [-10., 5., 10.],
    },
    Vertex {
        pos: [10., 5., -10.],
    },
    Vertex {
        pos: [10., 5., 10.],
    },
];
const INDICES_DATA: [VertexIndexType; 6] = [0, 2, 1, 1, 2, 3];

pub fn create_mesh() -> Mesh {
    let color = Color {
        color: [1.0, 1.0, 1.0],
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
