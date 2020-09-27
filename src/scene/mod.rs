use std::sync::Arc;

use cgmath::Point3;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::DescriptorSet;
use vulkano::device::{Device, Queue};
use vulkano::format::ClearValue;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::pipeline::GraphicsPipelineAbstract;

use crate::coords::SphericalPoint3;
use crate::mesh::Mesh;
use crate::scene::camera::Camera;
use crate::scene::lights::Lights;

mod baubles;
mod camera;
mod ground;
mod lights;

const CLEAR_VALUE: ClearValue = ClearValue::Float([0.015_7, 0., 0.360_7, 1.0]);

pub struct Scene {
    camera: Camera,
    lights: Lights,
    meshes: Vec<Mesh>,

    uniform_buffers: Arc<dyn DescriptorSet + Send + Sync>,
}

impl Scene {
    pub fn setup(
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        graphics_queue: Arc<Queue>,
        window_size: [u32; 2],
    ) -> Self {
        let camera = Self::setup_camera(window_size);
        let lights = Self::setup_lights();
        let meshes = Scene::setup_meshes(&graphics_queue);
        let uniform_buffers = Self::create_uniform_buffers(pipeline.clone(), graphics_queue.clone(), camera.clone(), lights.clone());
        Self { camera, lights, meshes, uniform_buffers }
    }

    fn setup_camera(window_size: [u32; 2]) -> Camera {
        let position: SphericalPoint3<f32> = SphericalPoint3::new(18., 1.7, 0.9);
        let look_at: Point3<f32> = Point3::new(0., 1., 0.);
        Camera::new(position, look_at, window_size)
    }

    fn setup_lights() -> Lights {
        let mut lights = Lights::setup();
        lights.add(Point3::new(10., -100., 10.), [0.3, 0.3, 0.3], [0.2, 0.2, 0.2], [0., 0., 0.]);
        lights.add(Point3::new(5., -6., 2.), [0.2, 0.2, 0.2], [2., 2., 2.], [0.5, 0.5, 0.5]);
        lights
    }

    fn setup_meshes(graphics_queue: &Arc<Queue>) -> Vec<Mesh> {
        let mut meshes: Vec<Mesh> = Vec::new();
        meshes.push(ground::create_mesh(graphics_queue.clone()));
        meshes.push(baubles::create_mesh(graphics_queue.clone()));
        meshes
    }

    fn create_uniform_buffers(
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        graphics_queue: Arc<Queue>,
        camera: Camera,
        lights: Lights,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let camera_buffer = camera.as_buffer(graphics_queue.clone());
        // let lights_buffer = lights.as_buffer(graphics_queue.clone());

        let layout = pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(camera_buffer)
                .unwrap()
                // .add_buffer(lights_buffer)
                // .unwrap()
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
