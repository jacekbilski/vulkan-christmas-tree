use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
use memoffset::offset_of;

use crate::mesh::Mesh;
use crate::scene::camera::Camera;
use crate::scene::lights::Lights;
use crate::vulkan::core::VulkanCore;
use crate::vulkan::graphics_execution::VulkanGraphicsExecution;
use crate::vulkan::graphics_setup::VulkanGraphicsSetup;

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
}

impl Vulkan {
    pub fn new(window: &winit::window::Window, application_name: &str) -> Self {
        let (core, surface_composite) = VulkanCore::new(&window, application_name);
        let graphics_setup = VulkanGraphicsSetup::new(core.clone(), surface_composite, &window);
        let graphics_execution = VulkanGraphicsExecution::new(core.clone(), &graphics_setup);

        Vulkan {
            core,
            graphics_setup,
            graphics_execution,
        }
    }

    pub fn set_meshes(&mut self, meshes: &Vec<Mesh>) {
        self.graphics_execution
            .set_meshes(meshes, &mut self.graphics_setup);
    }

    pub fn set_clear_value(&mut self, clear_value: [f32; 4]) {
        self.graphics_execution.set_clear_value(clear_value);
    }

    fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create Shader Module!")
        }
    }

    fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            ..Default::default()
        };

        unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create Command Pool!")
        }
    }

    fn find_supported_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        candidate_formats: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidate_formats.iter() {
            let format_properties =
                unsafe { instance.get_physical_device_format_properties(physical_device, format) };
            if tiling == vk::ImageTiling::LINEAR
                && format_properties.linear_tiling_features.contains(features)
            {
                return format.clone();
            } else if tiling == vk::ImageTiling::OPTIMAL
                && format_properties.optimal_tiling_features.contains(features)
            {
                return format.clone();
            }
        }

        panic!("Failed to find supported format!")
    }

    fn create_graphics_descriptor_pool(
        device: &ash::Device,
        swapchain_images_size: usize,
    ) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                // CameraUBO
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain_images_size as u32,
            },
            vk::DescriptorPoolSize {
                // LightsUBO
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain_images_size as u32,
            },
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            max_sets: swapchain_images_size as u32,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        }
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
        self.graphics_execution.draw_frame(&mut self.graphics_setup);
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
        self.graphics_execution.is_framebuffer_resized = true;
        self.graphics_setup.window_width = window_width;
        self.graphics_setup.window_height = window_height;
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        self.cleanup_swapchain();
        self.graphics_execution.drop();
        self.graphics_setup.drop();
        self.core.drop();
    }
}
