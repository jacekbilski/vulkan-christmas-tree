use std::sync::Arc;

use vulkano::buffer::{BufferAccess, TypedBufferAccess};

#[derive(Default, Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position);

#[derive(Default, Copy, Clone)]
pub struct InstanceData {
    pub colour: [f32; 3],
}
vulkano::impl_vertex!(InstanceData, colour);

pub struct Mesh {
    pub vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    pub index_buffer: Arc<dyn TypedBufferAccess<Content = [u32]> + Send + Sync>,
    pub instances_buffer: Arc<dyn BufferAccess + Send + Sync>,
}
