use std::sync::Arc;

use cgmath::Point3;
use vulkano::buffer::{BufferAccess, BufferUsage, ImmutableBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;

#[derive(Copy, Clone)]
struct Light {
    position: [f32; 3],
    ambient: [f32; 3],
    diffuse: [f32; 3],
    specular: [f32; 3],
}

pub struct Lights {
    lights: Vec<Light>,
}

impl Lights {
    pub fn setup() -> Self {
        Lights { lights: vec![] }
    }

    pub fn add(&mut self, position: Point3<f32>, ambient: [f32; 3], diffuse: [f32; 3], specular: [f32; 3]) {
        let light = Light {position: position.into(), ambient, diffuse, specular};
        self.lights.push(light);
    }

    pub fn as_buffer(&self, queue: Arc<Queue>) -> Arc<dyn BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_data(
            self.clone(), BufferUsage::uniform_buffer(), queue.clone()).unwrap();
        future.flush().unwrap();
        buffer
    }
}

impl Clone for Lights {
    fn clone(&self) -> Self {
        Lights { lights: self.lights.as_slice().clone().to_vec() }
    }
}
