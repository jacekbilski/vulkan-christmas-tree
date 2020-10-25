use cgmath::{Matrix4, SquareMatrix};

use crate::mesh::{Color, InstanceData, Mesh};
use crate::vulkan::Vertex;

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-0.5, 0.0, -0.5],
    },
    Vertex {
        pos: [0.5, 0.0, -0.5],
    },
    Vertex {
        pos: [0.5, 0.0, 0.5],
    },
    Vertex {
        pos: [-0.5, 0.0, 0.5],
    },
];
const INDICES_DATA: [u32; 6] = [0, 2, 1, 3, 2, 0];

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
