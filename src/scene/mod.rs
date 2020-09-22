use std::sync::Arc;

use vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuf};
use vulkano::device::{Device, Queue};
use vulkano::format::ClearValue;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::GraphicsPipelineAbstract;

use crate::Camera;
use crate::mesh::Mesh;

pub mod ground;

const CLEAR_VALUE: ClearValue = ClearValue::Float([0.015_7, 0., 0.360_7, 1.0]);

pub struct Scene {
    meshes: Vec<Mesh>,
}

impl Scene {
    pub fn setup(graphics_queue: Arc<Queue>) -> Self {
        let mut meshes: Vec<Mesh> = Vec::with_capacity(1);
        meshes.push(ground::create_mesh(graphics_queue));
        Self { meshes }
    }

    pub fn draw(
        &self,
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
        graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        dynamic_state: &DynamicState,
        uniform_buffers: Arc<PersistentDescriptorSet<((), PersistentDescriptorSetBuf<CpuBufferPoolSubbuffer<Camera, Arc<StdMemoryPool>>>)>>,
    ) -> AutoCommandBuffer {
        let mut command_builder = AutoCommandBufferBuilder::primary_one_time_submit(device, graphics_queue.family()).unwrap();
        command_builder
            .begin_render_pass(framebuffer, false, vec![CLEAR_VALUE])
            .unwrap()

            .draw_indexed(graphics_pipeline, dynamic_state, vec![self.meshes[0].vertex_buffer.clone(), self.meshes[0].instances_buffer.clone()], self.meshes[0].index_buffer.clone(), uniform_buffers, ())
            .unwrap()

            .end_render_pass()
            .unwrap();
        command_builder.build().unwrap()
    }
}
