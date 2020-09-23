use std::sync::Arc;

use cgmath::{Deg, Matrix4, perspective, Point3, vec3};
use vulkano::buffer::{BufferAccess, BufferUsage, ImmutableBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;

use crate::coords::SphericalPoint3;

#[derive(Copy, Clone)]
#[allow(unused)]
pub struct Camera {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl Camera {
    pub fn new(position: SphericalPoint3<f32>, look_at: Point3<f32>, window_size: [u32; 2]) -> Self {
        Camera {
            view: Matrix4::look_at(position.into(), look_at, vec3(0.0, 1.0, 0.0)),
            projection: perspective(Deg(45.0), window_size[0] as f32 / window_size[1] as f32, 0.1, 100.0),
        }
    }

    pub fn as_buffer(&self, queue: Arc<Queue>) -> Arc<dyn BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_data(
            self.clone(), BufferUsage::uniform_buffer(), queue.clone()).unwrap();
        future.flush().unwrap();
        buffer
    }
}
