use std::ffi::CString;
use std::ptr;

use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
use cgmath::{Matrix4, Point3};
use memoffset::offset_of;

use crate::fs::read_shader_code;
use crate::mesh::{InstanceData, Mesh};
use crate::scene::camera::Camera;
use crate::scene::lights::{Light, Lights};
use crate::vulkan::core::VulkanCore;
use crate::vulkan::graphics_setup::VulkanGraphicsSetup;

mod core;
mod graphics_setup;

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const CAMERA_UBO_INDEX: usize = 0;
const LIGHTS_UBO_INDEX: usize = 1;

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
}

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

#[derive(Clone)]
pub struct SwapChainComposite {
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
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

struct UniformBuffer {
    buffers: Vec<vk::Buffer>,              // one per swapchain_image_count
    buffers_memory: Vec<vk::DeviceMemory>, // one per swapchain_image_count
}

#[derive(Clone, Copy)]
struct VulkanMesh {
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    indices_no: u32,
    instance_buffer: vk::Buffer,
    instance_buffer_memory: vk::DeviceMemory,
    instances_no: u32,
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct CameraUBO {
    position: Point3<f32>,
    alignment_fix: f32, // see https://vulkan-tutorial.com/en/Uniform_buffers/Descriptor_pool_and_sets#page_Alignment-requirements
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

impl From<&Camera> for CameraUBO {
    fn from(camera: &Camera) -> Self {
        CameraUBO {
            position: camera.position.into(),
            alignment_fix: 0.0,
            view: camera.view,
            proj: camera.projection,
        }
    }
}

// TODO - how to handle layout 140 better?
#[repr(C)]
struct LightUBO {
    position: [f32; 3],
    alignment_fix_1: f32,
    ambient: [f32; 3],
    alignment_fix_2: f32,
    diffuse: [f32; 3],
    alignment_fix_3: f32,
    specular: [f32; 3],
    alignment_fix_4: f32,
}
impl From<Light> for LightUBO {
    fn from(light: Light) -> Self {
        LightUBO {
            position: light.position,
            ambient: light.ambient,
            diffuse: light.diffuse,
            specular: light.specular,
            alignment_fix_1: 0.0,
            alignment_fix_2: 0.0,
            alignment_fix_3: 0.0,
            alignment_fix_4: 0.0,
        }
    }
}

#[repr(C)]
struct LightsUBO {
    count: u32,
    alignment_fix_1: [f32; 3],
    lights: [LightUBO; 2], // hardcoded "2"
}

impl From<&Lights> for LightsUBO {
    fn from(lights: &Lights) -> Self {
        LightsUBO {
            count: lights.lights.len() as u32,
            alignment_fix_1: [0., 0., 0.],
            lights: [
                LightUBO::from(lights.lights[0]),
                LightUBO::from(lights.lights[1]),
            ],
        }
    }
}

struct VulkanGraphicsExecution {
    clear_value: [f32; 4],

    uniform_buffers: Vec<UniformBuffer>,
    meshes: Vec<VulkanMesh>,
    graphics_descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,
}

impl VulkanGraphicsExecution {
    fn new(core: &VulkanCore, graphics_setup: &VulkanGraphicsSetup) -> Self {
        let uniform_buffers = VulkanGraphicsExecution::create_uniform_buffers(
            &core,
            graphics_setup.swapchain_composite.images.len(),
        );
        let graphics_descriptor_sets = VulkanGraphicsExecution::create_graphics_descriptor_sets(
            &core.device,
            graphics_setup.descriptor_pool,
            graphics_setup.descriptor_set_layout,
            &uniform_buffers,
            graphics_setup.swapchain_composite.images.len(),
        );
        let sync_objects = VulkanGraphicsExecution::create_sync_objects(&core.device);

        VulkanGraphicsExecution {
            clear_value: [0.0, 0.0, 0.0, 0.0],

            uniform_buffers,
            meshes: vec![],
            graphics_descriptor_sets,
            command_buffers: vec![],

            image_available_semaphores: sync_objects.image_available_semaphores,
            render_finished_semaphores: sync_objects.render_finished_semaphores,
            in_flight_fences: sync_objects.inflight_fences,
            current_frame: 0,

            is_framebuffer_resized: false,
        }
    }

    fn create_uniform_buffers(
        core: &VulkanCore,
        swapchain_image_count: usize,
    ) -> Vec<UniformBuffer> {
        let mut uniform_buffers = vec![];

        {
            let buffer_size = std::mem::size_of::<CameraUBO>();

            let mut buffers = vec![];
            let mut buffers_memory = vec![];

            for _ in 0..swapchain_image_count {
                let (uniform_buffer, uniform_buffer_memory) = Vulkan::create_buffer(
                    core,
                    buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                buffers.push(uniform_buffer);
                buffers_memory.push(uniform_buffer_memory);
            }

            uniform_buffers.push(UniformBuffer {
                buffers,
                buffers_memory,
            });
        }
        {
            let buffer_size = std::mem::size_of::<LightsUBO>();

            let mut buffers = vec![];
            let mut buffers_memory = vec![];

            for _ in 0..swapchain_image_count {
                let (uniform_buffer, uniform_buffer_memory) = Vulkan::create_buffer(
                    core,
                    buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                buffers.push(uniform_buffer);
                buffers_memory.push(uniform_buffer_memory);
            }

            uniform_buffers.push(UniformBuffer {
                buffers,
                buffers_memory,
            });
        }
        uniform_buffers
    }

    fn create_graphics_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniforms_buffers: &Vec<UniformBuffer>,
        swapchain_images_size: usize,
    ) -> Vec<vk::DescriptorSet> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..swapchain_images_size {
            layouts.push(descriptor_set_layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: swapchain_images_size as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        for (i, &descritptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_buffer_info = [
                vk::DescriptorBufferInfo {
                    buffer: uniforms_buffers[CAMERA_UBO_INDEX].buffers[i],
                    offset: 0,
                    range: std::mem::size_of::<CameraUBO>() as u64,
                },
                vk::DescriptorBufferInfo {
                    buffer: uniforms_buffers[LIGHTS_UBO_INDEX].buffers[i],
                    offset: 0,
                    range: std::mem::size_of::<LightsUBO>() as u64,
                },
            ];

            let descriptor_write_sets = [vk::WriteDescriptorSet {
                dst_set: descritptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: descriptor_buffer_info.len() as u32,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: descriptor_buffer_info.as_ptr(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            }];

            unsafe {
                device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
        }

        descriptor_sets
    }

    fn create_sync_objects(device: &ash::Device) -> SyncObjects {
        let mut sync_objects = SyncObjects {
            image_available_semaphores: vec![],
            render_finished_semaphores: vec![],
            inflight_fences: vec![],
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };

        let fence_create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let inflight_fence = device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create Fence Object!");

                sync_objects
                    .image_available_semaphores
                    .push(image_available_semaphore);
                sync_objects
                    .render_finished_semaphores
                    .push(render_finished_semaphore);
                sync_objects.inflight_fences.push(inflight_fence);
            }
        }

        sync_objects
    }

    fn drop(&mut self, device: &ash::Device) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                device.destroy_semaphore(self.image_available_semaphores[i], None);
                device.destroy_semaphore(self.render_finished_semaphores[i], None);
                device.destroy_fence(self.in_flight_fences[i], None);
            }

            for mesh in self.meshes.iter() {
                device.destroy_buffer(mesh.instance_buffer, None);
                device.free_memory(mesh.instance_buffer_memory, None);
                device.destroy_buffer(mesh.index_buffer, None);
                device.free_memory(mesh.index_buffer_memory, None);
                device.destroy_buffer(mesh.vertex_buffer, None);
                device.free_memory(mesh.vertex_buffer_memory, None);
            }
            for j in 0..self.uniform_buffers.len() {
                for i in 0..self.uniform_buffers[j].buffers.len() {
                    device.destroy_buffer(self.uniform_buffers[j].buffers[i], None);
                    device.free_memory(self.uniform_buffers[j].buffers_memory[i], None);
                }
            }
        }
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
        let graphics_setup = VulkanGraphicsSetup::new(&core, surface_composite, &window);
        let graphics_execution = VulkanGraphicsExecution::new(&core, &graphics_setup);

        Vulkan {
            core,
            graphics_setup,
            graphics_execution,
        }
    }

    pub fn set_meshes(&mut self, meshes: &Vec<Mesh>) {
        let mut vulkan_meshes: Vec<VulkanMesh> = vec![];

        for mesh in meshes.iter() {
            let (vertex_buffer, vertex_buffer_memory) = Vulkan::create_vertex_buffer(
                &self.core,
                self.graphics_setup.command_pool,
                &mesh.vertices,
            );
            let (index_buffer, index_buffer_memory) = Vulkan::create_index_buffer(
                &self.core,
                self.graphics_setup.command_pool,
                &mesh.indices,
            );
            let indices_no = mesh.indices.len() as u32;
            let (instance_buffer, instance_buffer_memory) = Vulkan::create_vertex_buffer(
                &self.core,
                self.graphics_setup.command_pool,
                &mesh.instances,
            );
            let instances_no = mesh.instances.len() as u32;
            vulkan_meshes.push(VulkanMesh {
                vertex_buffer,
                vertex_buffer_memory,
                index_buffer,
                index_buffer_memory,
                indices_no,
                instance_buffer,
                instance_buffer_memory,
                instances_no,
            });
        }
        let command_buffers = Vulkan::create_command_buffers(
            &self.core.device,
            &self.graphics_setup,
            &vulkan_meshes,
            &self.graphics_execution.graphics_descriptor_sets,
            self.graphics_execution.clear_value,
        );
        self.graphics_execution.meshes = vulkan_meshes;
        self.graphics_execution.command_buffers = command_buffers;
    }

    pub fn set_clear_value(&mut self, clear_value: [f32; 4]) {
        self.graphics_execution.clear_value = clear_value;
    }

    fn create_render_pass(core: &VulkanCore, surface_format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };

        let depth_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: Vulkan::find_depth_format(&core.instance, core.physical_device),
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpasses = [vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            preserve_attachment_count: 0,
            ..Default::default()
        }];

        let render_pass_attachments = [color_attachment, depth_attachment];

        let subpass_dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];

        let renderpass_create_info = vk::RenderPassCreateInfo {
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr(),
            ..Default::default()
        };

        unsafe {
            core.device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create render pass!")
        }
    }

    fn create_graphics_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let descriptor_set_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: CAMERA_UBO_INDEX as u32,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: LIGHTS_UBO_INDEX as u32,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: descriptor_set_layout_bindings.len() as u32,
            p_bindings: descriptor_set_layout_bindings.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create Descriptor Set Layout!")
        }
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_shader_code = read_shader_code("simple.vert.spv");
        let frag_shader_code = read_shader_code("simple.frag.spv");

        let vert_shader_module = Vulkan::create_shader_module(device, vert_shader_code);
        let frag_shader_module = Vulkan::create_shader_module(device, frag_shader_code);

        let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                module: vert_shader_module,
                p_name: main_function_name.as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                module: frag_shader_module,
                p_name: main_function_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let mut binding_description: Vec<vk::VertexInputBindingDescription> = vec![];
        for &bd in Vertex::get_binding_descriptions().iter() {
            binding_description.push(bd);
        }
        for &bd in InstanceData::get_binding_descriptions().iter() {
            binding_description.push(bd);
        }
        let mut attribute_description: Vec<vk::VertexInputAttributeDescription> = vec![];
        for &ad in Vertex::get_attribute_descriptions().iter() {
            attribute_description.push(ad);
        }
        for &ad in InstanceData::get_attribute_descriptions().iter() {
            attribute_description.push(ad);
        }

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_attribute_description_count: attribute_description.len() as u32,
            p_vertex_attribute_descriptions: attribute_description.as_ptr(),
            vertex_binding_description_count: binding_description.len() as u32,
            p_vertex_binding_descriptions: binding_description.as_ptr(),
            ..Default::default()
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            primitive_restart_enable: vk::FALSE,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
            ..Default::default()
        };

        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_clamp: 0.0,
            depth_bias_constant_factor: 0.0,
            depth_bias_enable: vk::FALSE,
            depth_bias_slope_factor: 0.0,
            ..Default::default()
        };
        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            alpha_to_one_enable: vk::FALSE,
            alpha_to_coverage_enable: vk::FALSE,
            ..Default::default()
        };

        let stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        };

        let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: stencil_state,
            back: stencil_state,
            max_depth_bounds: 1.0,
            min_depth_bounds: 0.0,
            ..Default::default()
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::all(),
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        }];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };

        let set_layouts = [descriptor_set_layout];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state_create_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_create_info,
            p_rasterization_state: &rasterization_state_create_info,
            p_multisample_state: &multisample_state_create_info,
            p_depth_stencil_state: &depth_state_create_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: ptr::null(),
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            ..Default::default()
        }];

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &graphic_pipeline_create_infos,
                    None,
                )
                .expect("Failed to create Graphics Pipeline!.")
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        (graphics_pipelines[0], pipeline_layout)
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

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        image_views: &Vec<vk::ImageView>,
        depth_image_view: vk::ImageView,
        swapchain_extent: &vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = vec![];

        for &image_view in image_views.iter() {
            let attachments = [image_view, depth_image_view];

            let framebuffer_create_info = vk::FramebufferCreateInfo {
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                layers: 1,
                ..Default::default()
            };

            let framebuffer = unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create Framebuffer!")
            };

            framebuffers.push(framebuffer);
        }

        framebuffers
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

    fn create_depth_resources(
        core: &VulkanCore,
        swapchain_extent: vk::Extent2D,
    ) -> (vk::Image, vk::ImageView, vk::DeviceMemory) {
        let depth_format = Vulkan::find_depth_format(&core.instance, core.physical_device);
        let (depth_image, depth_image_memory) = Vulkan::create_image(
            &core.device,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            vk::SampleCountFlags::TYPE_1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &core.physical_device_memory_properties,
        );
        let depth_image_view = Vulkan::create_image_view(
            &core.device,
            depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
            1,
        );

        (depth_image, depth_image_view, depth_image_memory)
    }

    fn find_depth_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> vk::Format {
        Vulkan::find_supported_format(
            instance,
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
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
        let ubo: CameraUBO = CameraUBO::from(camera);
        let ubos = [ubo];

        let buffer_size = (std::mem::size_of::<CameraUBO>() * ubos.len()) as u64;

        for current_image in 0..self.graphics_setup.swapchain_composite.images.len() {
            unsafe {
                let data_ptr =
                    self.core
                        .device
                        .map_memory(
                            self.graphics_execution.uniform_buffers[CAMERA_UBO_INDEX]
                                .buffers_memory[current_image],
                            0,
                            buffer_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to Map Memory") as *mut CameraUBO;

                data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

                self.core.device.unmap_memory(
                    self.graphics_execution.uniform_buffers[CAMERA_UBO_INDEX].buffers_memory
                        [current_image],
                );
            }
        }
    }

    pub fn update_lights(&mut self, lights: &Lights) {
        let ubo: LightsUBO = LightsUBO::from(lights);
        let ubos = [ubo];

        let buffer_size = (std::mem::size_of::<LightsUBO>() * ubos.len()) as u64;

        for current_image in 0..self.graphics_setup.swapchain_composite.images.len() {
            unsafe {
                let data_ptr =
                    self.core
                        .device
                        .map_memory(
                            self.graphics_execution.uniform_buffers[LIGHTS_UBO_INDEX]
                                .buffers_memory[current_image],
                            0,
                            buffer_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to Map Memory") as *mut LightsUBO;

                data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

                self.core.device.unmap_memory(
                    self.graphics_execution.uniform_buffers[LIGHTS_UBO_INDEX].buffers_memory
                        [current_image],
                );
            }
        }
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

    fn create_command_buffers(
        device: &ash::Device,
        graphics_setup: &VulkanGraphicsSetup,
        meshes: &Vec<VulkanMesh>,
        descriptor_sets: &Vec<vk::DescriptorSet>,
        clear_value: [f32; 4],
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: graphics_setup.swapchain_composite.framebuffers.len() as u32,
            command_pool: graphics_setup.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        };

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                p_inheritance_info: ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
                ..Default::default()
            };

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: clear_value,
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                render_pass: graphics_setup.render_pass,
                framebuffer: graphics_setup.swapchain_composite.framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: graphics_setup.swapchain_composite.extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
                ..Default::default()
            };

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_setup.pipeline,
                );

                let descriptor_sets_to_bind = [descriptor_sets[i]];
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_setup.pipeline_layout,
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                for mesh in meshes.iter() {
                    let vertex_buffers = [mesh.vertex_buffer, mesh.instance_buffer];
                    let offsets = [0_u64, 0_u64];

                    device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                    device.cmd_bind_index_buffer(
                        command_buffer,
                        mesh.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(
                        command_buffer,
                        mesh.indices_no,
                        mesh.instances_no,
                        0,
                        0,
                        0,
                    );
                }

                device.cmd_end_render_pass(command_buffer);

                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        command_buffers
    }

    pub fn draw_frame(&mut self) {
        let wait_fences =
            [self.graphics_execution.in_flight_fences[self.graphics_execution.current_frame]];

        unsafe {
            self.core
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self
                .graphics_setup
                .swapchain_composite
                .loader
                .acquire_next_image(
                    self.graphics_setup.swapchain_composite.swapchain,
                    std::u64::MAX,
                    self.graphics_execution.image_available_semaphores
                        [self.graphics_execution.current_frame],
                    vk::Fence::null(),
                );
            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain();
                        return;
                    }
                    _ => panic!("Failed to acquire Swap Chain Image!"),
                },
            }
        };

        let wait_semaphores = [self.graphics_execution.image_available_semaphores
            [self.graphics_execution.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.graphics_execution.render_finished_semaphores
            [self.graphics_execution.current_frame]];

        let submit_infos = [vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.graphics_execution.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            self.core
                .device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.core
                .device
                .queue_submit(
                    self.core.graphics_queue,
                    &submit_infos,
                    self.graphics_execution.in_flight_fences[self.graphics_execution.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.graphics_setup.swapchain_composite.swapchain];

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };

        let result = unsafe {
            self.graphics_setup
                .swapchain_composite
                .loader
                .queue_present(self.core.present_queue, &present_info)
        };
        let is_resized = match result {
            Ok(_) => self.graphics_execution.is_framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.graphics_execution.is_framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.graphics_execution.current_frame =
            (self.graphics_execution.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain(&mut self) {
        self.graphics_setup.recreate_swapchain(&self.core);
        self.graphics_execution.command_buffers = Vulkan::create_command_buffers(
            &self.core.device,
            &self.graphics_setup,
            &self.graphics_execution.meshes,
            &self.graphics_execution.graphics_descriptor_sets,
            self.graphics_execution.clear_value,
        );
    }

    fn cleanup_swapchain(&self, device: &ash::Device) {
        unsafe {
            device.free_command_buffers(
                self.graphics_setup.command_pool,
                &self.graphics_execution.command_buffers,
            );
            self.graphics_setup.cleanup_swapchain(&device);
        }
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
        let device = &self.core.device;
        self.cleanup_swapchain(&device);
        self.graphics_execution.drop(&device);
        self.graphics_setup.drop(&device);
        self.core.drop();
    }
}
