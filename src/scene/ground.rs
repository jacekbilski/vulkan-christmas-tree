use std::sync::Arc;

use cgmath::{Matrix4, SquareMatrix};
use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;

use crate::mesh::{InstanceData, Mesh, Vertex, VertexIndexType};

pub fn create_mesh(queue: Arc<Queue>) -> Mesh {
    let vertices: Vec<Vertex> = vec![
        Vertex { position: [-10., 5., -10.], normal: [0., -1., 0.] },   // far
        Vertex { position: [-10., 5., 10.], normal: [0., -1., 0.] }, // left
        Vertex { position: [10., 5., -10.], normal: [0., -1., 0.] }, // right
        Vertex { position: [10., 5., 10.], normal: [0., -1., 0.] }, // near
    ];
    let (vertex_buffer, vertex_future) = ImmutableBuffer::from_iter(
        vertices.into_iter(), BufferUsage::vertex_buffer(), queue.clone())
        .unwrap();

    let indices: Vec<VertexIndexType> = vec![
        0, 1, 2,
        1, 3, 2,
    ];

    let (index_buffer, index_future) = ImmutableBuffer::from_iter(
        indices.iter().cloned(), BufferUsage::index_buffer(), queue.clone()).unwrap();

    let instances: Vec<InstanceData> = vec![
        InstanceData { model: Matrix4::identity().into(), colour: [1.0, 1.0, 1.0] }
    ];

    let (instances_buffer, instances_future) = ImmutableBuffer::from_iter(
        instances.into_iter(), BufferUsage::vertex_buffer(), queue.clone())
        .unwrap();

    vertex_future.flush().unwrap();
    index_future.flush().unwrap();
    instances_future.flush().unwrap();
    Mesh { vertex_buffer, index_buffer, instances_buffer }
}
