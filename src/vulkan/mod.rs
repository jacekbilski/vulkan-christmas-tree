use std::ptr;

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

    fn create_image(
        device: &ash::Device,
        width: u32,
        height: u32,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            mip_levels,
            array_layers: 1,
            samples: num_samples,
            tiling,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ..Default::default()
        };

        let image = unsafe {
            device
                .create_image(&image_create_info, None)
                .expect("Failed to create Texture Image!")
        };

        let image_memory_requirement = unsafe { device.get_image_memory_requirements(image) };
        let memory_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: image_memory_requirement.size,
            memory_type_index: Vulkan::find_memory_type(
                image_memory_requirement.memory_type_bits,
                required_memory_properties,
                device_memory_properties,
            ),
            ..Default::default()
        };

        let image_memory = unsafe {
            device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate Texture Image memory!")
        };

        unsafe {
            device
                .bind_image_memory(image, image_memory, 0)
                .expect("Failed to bind Image Memmory!");
        }

        (image, image_memory)
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

    fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let imageview_create_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            image,
            ..Default::default()
        };

        unsafe {
            device
                .create_image_view(&imageview_create_info, None)
                .expect("Failed to create Image View!")
        }
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

    fn create_vertex_buffer<T>(
        core: &VulkanCore,
        command_pool: vk::CommandPool,
        data: &[T],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Vulkan::create_buffer(
            core,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = core
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut T;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

            core.device.unmap_memory(staging_buffer_memory);
        }

        let (vertex_buffer, vertex_buffer_memory) = Vulkan::create_buffer(
            core,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Vulkan::copy_buffer(
            &core,
            command_pool,
            staging_buffer,
            vertex_buffer,
            buffer_size,
        );

        unsafe {
            core.device.destroy_buffer(staging_buffer, None);
            core.device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_buffer(
        core: &VulkanCore,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            ..Default::default()
        };

        let buffer = unsafe {
            core.device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create Vertex Buffer")
        };

        let mem_requirements = unsafe { core.device.get_buffer_memory_requirements(buffer) };
        let memory_type = Vulkan::find_memory_type(
            mem_requirements.memory_type_bits,
            required_memory_properties,
            &core.physical_device_memory_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let buffer_memory = unsafe {
            core.device
                .allocate_memory(&allocate_info, None)
                .expect("Failed to allocate vertex buffer memory!")
        };

        unsafe {
            core.device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind Buffer");
        }

        (buffer, buffer_memory)
    }

    fn find_memory_type(
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
        mem_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
            // same implementation
            if (type_filter & (1 << i)) > 0
                && memory_type.property_flags.contains(required_properties)
            {
                return i as u32;
            }
        }

        panic!("Failed to find suitable memory type!")
    }

    fn copy_buffer(
        core: &VulkanCore,
        command_pool: vk::CommandPool,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: 1,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffers = unsafe {
            core.device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate Command Buffer")
        };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        unsafe {
            core.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin Command Buffer");

            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }];

            core.device
                .cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);

            core.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end Command Buffer");
        }

        let submit_info = [vk::SubmitInfo {
            wait_semaphore_count: 0,
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            ..Default::default()
        }];

        unsafe {
            core.device
                .queue_submit(core.transfer_queue, &submit_info, vk::Fence::null())
                .expect("Failed to Submit Queue.");
            core.device
                .queue_wait_idle(core.transfer_queue)
                .expect("Failed to wait Queue idle");

            core.device
                .free_command_buffers(command_pool, &command_buffers);
        }
    }

    fn create_index_buffer(
        core: &VulkanCore,
        command_pool: vk::CommandPool,
        data: &[u32],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Vulkan::create_buffer(
            core,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = core
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut u32;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

            core.device.unmap_memory(staging_buffer_memory);
        }

        let (index_buffer, index_buffer_memory) = Vulkan::create_buffer(
            core,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Vulkan::copy_buffer(
            &core,
            command_pool,
            staging_buffer,
            index_buffer,
            buffer_size,
        );

        unsafe {
            core.device.destroy_buffer(staging_buffer, None);
            core.device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
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
