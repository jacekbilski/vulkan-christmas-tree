use ash::version::DeviceV1_0;
use ash::vk;
use memoffset::offset_of;

use crate::mesh::{InstanceData, Mesh};
use crate::scene::camera::Camera;
use crate::scene::lights::Lights;
use crate::scene::snow::MAX_SNOWFLAKES;
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
    loader: ash::extensions::khr::Surface,
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
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
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

    pub fn set_meshes(&mut self, meshes: &Vec<Mesh>) {
        let (buffer, _buffer_memory) = self
            .graphics_execution
            .set_meshes(meshes, &mut self.graphics_setup);

        self.compute_execution = Some(VulkanComputeExecution::new(
            self.core.clone(),
            &self.compute_setup,
            buffer,
            std::mem::size_of::<InstanceData>() * MAX_SNOWFLAKES,
        ));
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

    pub fn draw_frame(&mut self) {
        self.compute_execution
            .as_ref()
            .unwrap()
            .do_calculations(self.snow_calculated_semaphore);
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
