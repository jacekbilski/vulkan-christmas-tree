use std::sync::Arc;

use vulkano::buffer::{BufferAccess, TypedBufferAccess};

pub(crate) type VertexIndexType = u32;

#[derive(Default, Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal);

#[derive(Default, Copy, Clone)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
    pub shininess: f32,
}
vulkano::impl_vertex!(InstanceData, model, ambient, diffuse, specular, shininess);

pub struct Mesh {
    pub vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    pub index_buffer: Arc<dyn TypedBufferAccess<Content = [VertexIndexType]> + Send + Sync>,
    pub instances_buffer: Arc<dyn BufferAccess + Send + Sync>,
}
