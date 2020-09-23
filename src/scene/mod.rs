use std::sync::Arc;

use cgmath::{Deg, Matrix4, perspective, Point3, vec3};
use vulkano::buffer::{BufferAccess, BufferUsage, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuf};
use vulkano::device::{Device, Queue};
use vulkano::format::ClearValue;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::sync::GpuFuture;

use crate::coords::SphericalPoint3;
use crate::mesh::Mesh;

mod baubles;
mod ground;

const CLEAR_VALUE: ClearValue = ClearValue::Float([0.015_7, 0., 0.360_7, 1.0]);

#[derive(Copy, Clone)]
#[allow(unused)]
struct Camera {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl Camera {
    fn new(position: SphericalPoint3<f32>, look_at: Point3<f32>, window_size: [u32; 2]) -> Self {
        Camera {
            view: Matrix4::look_at(position.into(), look_at, vec3(0.0, 1.0, 0.0)),
            projection: perspective(Deg(45.0), window_size[0] as f32 / window_size[1] as f32, 0.1, 100.0),
        }
    }

    fn as_buffer(&self, queue: Arc<Queue>) -> Arc<dyn BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_data(
            self.clone(), BufferUsage::uniform_buffer(), queue.clone()).unwrap();
        future.flush().unwrap();
        buffer
    }
}

pub struct Scene {
    camera: Camera,
    meshes: Vec<Mesh>,

    // this should be of type Arc<dyn DescriptorSetsCollection + Send + Sync>
    uniform_buffers: Arc<PersistentDescriptorSet<((), PersistentDescriptorSetBuf<Arc<dyn BufferAccess + Send + Sync>>)>>,
}

impl Scene {
    pub fn setup(
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        graphics_queue: Arc<Queue>,
        window_size: [u32; 2],
    ) -> Self {
        let position: SphericalPoint3<f32> = SphericalPoint3::new(18., 1.7, 0.9);
        let look_at: Point3<f32> = Point3::new(0., 1., 0.);
        let camera = Camera::new(position, look_at, window_size);

        let mut meshes: Vec<Mesh> = Vec::with_capacity(1);
        meshes.push(ground::create_mesh(graphics_queue.clone()));
        meshes.push(baubles::create_mesh(graphics_queue.clone()));
        let uniform_buffers = Self::create_uniform_buffers(pipeline.clone(), graphics_queue.clone(), camera.clone());
        Self { camera, meshes, uniform_buffers }
    }

    fn create_uniform_buffers(
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        graphics_queue: Arc<Queue>,
        camera: Camera,
    ) -> Arc<PersistentDescriptorSet<((), PersistentDescriptorSetBuf<Arc<dyn BufferAccess + Send + Sync>>)>> {
        let camera_buffer = camera.as_buffer(graphics_queue.clone());

        let layout = pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(camera_buffer)
                .unwrap()
                .build()
                .unwrap()
        );
        set
    }

    pub fn draw(
        &self,
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
        graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        dynamic_state: &DynamicState,
    ) -> AutoCommandBuffer {
        let mut command_builder = AutoCommandBufferBuilder::primary_one_time_submit(device, graphics_queue.family()).unwrap();
        command_builder
            .begin_render_pass(framebuffer, false, vec![CLEAR_VALUE])
            .unwrap();

        for mesh in &self.meshes[..] {
            command_builder
                .draw_indexed(
                    graphics_pipeline.clone(),
                    dynamic_state,
                    vec![mesh.vertex_buffer.clone(), mesh.instances_buffer.clone()],
                    mesh.index_buffer.clone(),
                    self.uniform_buffers.clone(),
                    ())
                .unwrap();
        }

        command_builder
            .end_render_pass()
            .unwrap();
        command_builder.build().unwrap()
    }
}
