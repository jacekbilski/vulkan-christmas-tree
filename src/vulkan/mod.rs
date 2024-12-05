use ash::{khr, vk};
use memoffset::offset_of;

use crate::color_mesh::{ColorMesh, InstanceData};
use crate::scene::camera::Camera;
use crate::scene::lights::Lights;
use crate::scene::snow::{Snowflake, MAX_SNOWFLAKES};
use crate::textured_mesh::TexturedMesh;
use crate::vulkan::compute_execution::VulkanComputeExecution;
use crate::vulkan::compute_setup::VulkanComputeSetup;
use crate::vulkan::core::VulkanCore;
use crate::vulkan::graphics_execution::VulkanGraphicsExecution;
use crate::vulkan::graphics_setup::VulkanGraphicsSetup;

mod compute_execution;
mod compute_setup;
mod core;
mod graphics_execution;
mod graphics_setup;

#[derive(Clone)]
pub struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    compute_family: Option<u32>,
    transfer_family: Option<u32>,
    present_family: Option<u32>,
}
impl QueueFamilyIndices {
    fn new() -> Self {
        Self {
            graphics_family: None,
            compute_family: None,
            transfer_family: None,
            present_family: None,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
            && self.compute_family.is_some()
            && self.transfer_family.is_some()
            && self.present_family.is_some()
    }
}

pub struct SurfaceComposite {
    loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

pub(crate) type VertexIndexType = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub norm: [f32; 3],
}
impl Vertex {
    fn get_binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT, // aka vec3
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT, // aka vec3
                offset: offset_of!(Self, norm) as u32,
            },
        ]
    }
}

pub struct Vulkan {
    core: VulkanCore,
    graphics_setup: VulkanGraphicsSetup,
    graphics_execution: VulkanGraphicsExecution,
    compute_setup: VulkanComputeSetup,
    compute_execution: Option<VulkanComputeExecution>,

    snow_calculated_semaphore: vk::Semaphore,
}

impl Vulkan {
    pub fn new(window: &winit::window::Window, application_name: &str) -> Self {
        let (core, surface_composite) = VulkanCore::new(&window, application_name);
        let graphics_setup = VulkanGraphicsSetup::new(core.clone(), surface_composite, &window);
        let graphics_execution = VulkanGraphicsExecution::new(core.clone(), &graphics_setup);
        let compute_setup = VulkanComputeSetup::new(core.clone());

        let snow_calculated_semaphore = core.create_semaphore();

        Vulkan {
            core,
            graphics_setup,
            graphics_execution,
            compute_setup,
            compute_execution: None,

            snow_calculated_semaphore,
        }
    }

    pub fn set_static_meshes(
        &mut self,
        color_meshes: &Vec<ColorMesh>,
        textured_meshes: &Vec<TexturedMesh>,
    ) {
        self.graphics_execution.set_static_meshes(
            color_meshes,
            textured_meshes,
            &mut self.graphics_setup,
        );
    }

    pub fn set_snow_mesh(&mut self, snowflakes: &Vec<Snowflake>, meshes: &Vec<ColorMesh>) {
        let (drawing_buffer, _buffer_memory) = self
            .graphics_execution
            .set_snow_mesh(meshes, &mut self.graphics_setup);

        self.compute_execution = Some(VulkanComputeExecution::new(
            self.core.clone(),
            self.compute_setup.clone(),
            snowflakes,
            drawing_buffer,
            size_of::<InstanceData>() * MAX_SNOWFLAKES,
        ));
    }

    pub fn scene_complete(&mut self) {
        self.graphics_execution
            .create_command_buffers(&self.graphics_setup);
    }

    pub fn set_clear_value(&mut self, clear_value: [f32; 4]) {
        self.graphics_execution.set_clear_value(clear_value);
    }

    pub fn update_camera(&mut self, camera: &Camera) {
        self.graphics_execution
            .update_camera(camera, &self.graphics_setup);
    }

    pub fn update_lights(&mut self, lights: &Lights) {
        self.graphics_execution
            .update_lights(lights, &self.graphics_setup);
    }

    pub fn draw_frame(&mut self, last_frame_time_secs: f32) {
        self.compute_execution
            .as_mut()
            .unwrap()
            .do_calculations(self.snow_calculated_semaphore, last_frame_time_secs);
        self.graphics_execution
            .draw_frame(&mut self.graphics_setup, self.snow_calculated_semaphore);
    }

    fn cleanup_swapchain(&self) {
        self.graphics_execution
            .cleanup_swapchain(self.graphics_setup.command_pool);
        self.graphics_setup.cleanup_swapchain();
    }

    pub fn wait_device_idle(&self) {
        unsafe {
            self.core
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

    pub fn framebuffer_resized(&mut self, window_width: u32, window_height: u32) {
        self.graphics_execution.framebuffer_resized();
        self.graphics_setup
            .framebuffer_resized(window_width, window_height);
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device
                .destroy_semaphore(self.snow_calculated_semaphore, None);
        }
        if self.compute_execution.is_some() {
            self.compute_execution
                .as_ref()
                .unwrap()
                .drop(&self.compute_setup);
        }
        self.compute_setup.drop();
        self.cleanup_swapchain();
        self.graphics_execution.drop();
        self.graphics_setup.drop();
        self.core.drop();
    }
}
