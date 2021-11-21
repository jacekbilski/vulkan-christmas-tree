use std::ffi::CString;
use std::ptr;

use ash::vk;

use crate::mesh::InstanceData;
use crate::vulkan::core::VulkanCore;
use crate::vulkan::{SurfaceComposite, Vertex};

pub const CAMERA_UBO_INDEX: usize = 0;
pub const LIGHTS_UBO_INDEX: usize = 1;

const COLOR_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

#[derive(Clone)]
pub struct SwapChainComposite {
    pub loader: ash::extensions::khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    format: vk::Format,
    pub extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
}

pub struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct VulkanGraphicsSetup {
    core: VulkanCore,

    surface_composite: SurfaceComposite,
    pub swapchain_composite: SwapChainComposite,

    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    msaa_samples: vk::SampleCountFlags,

    color_image: vk::Image,
    color_image_view: vk::ImageView,
    color_image_memory: vk::DeviceMemory,

    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,

    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,

    window_width: u32,
    window_height: u32,
}

impl VulkanGraphicsSetup {
    pub fn new(
        core: VulkanCore,
        surface_composite: SurfaceComposite,
        window: &winit::window::Window,
    ) -> Self {
        let window_width = window.inner_size().width;
        let window_height = window.inner_size().height;
        let mut swapchain_composite = VulkanGraphicsSetup::create_swapchain(
            &core,
            &surface_composite,
            window_width,
            window_height,
        );
        swapchain_composite.image_views =
            VulkanGraphicsSetup::create_image_views(&core.device, &swapchain_composite);
        let msaa_samples = VulkanGraphicsSetup::choose_msaa_samples(&core);
        let render_pass = VulkanGraphicsSetup::create_render_pass(
            &core,
            swapchain_composite.format,
            msaa_samples,
        );
        let descriptor_set_layout = VulkanGraphicsSetup::create_descriptor_set_layout(&core.device);
        let (pipeline, pipeline_layout) = VulkanGraphicsSetup::create_pipeline(
            &core,
            render_pass,
            swapchain_composite.extent,
            descriptor_set_layout,
            msaa_samples,
        );
        let (color_image, color_image_view, color_image_memory) =
            VulkanGraphicsSetup::create_color_resources(
                &core,
                swapchain_composite.extent,
                msaa_samples,
            );
        let (depth_image, depth_image_view, depth_image_memory) =
            VulkanGraphicsSetup::create_depth_resources(
                &core,
                swapchain_composite.extent,
                msaa_samples,
            );
        swapchain_composite.framebuffers = VulkanGraphicsSetup::create_framebuffers(
            &core.device,
            render_pass,
            &swapchain_composite.image_views,
            color_image_view,
            depth_image_view,
            &swapchain_composite.extent,
        );
        let command_pool = core.create_command_pool(core.queue_family.graphics_family.unwrap());
        let descriptor_pool = VulkanGraphicsSetup::create_descriptor_pool(
            &core.device,
            swapchain_composite.images.len(),
        );

        VulkanGraphicsSetup {
            core,

            surface_composite,
            swapchain_composite,

            render_pass,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,

            msaa_samples,

            color_image,
            color_image_view,
            color_image_memory,

            depth_image,
            depth_image_view,
            depth_image_memory,

            command_pool,
            descriptor_pool,

            window_width,
            window_height,
        }
    }

    fn create_swapchain(
        core: &VulkanCore,
        surface_composite: &SurfaceComposite,
        window_width: u32,
        window_height: u32,
    ) -> SwapChainComposite {
        let swapchain_support =
            VulkanGraphicsSetup::find_swapchain_support(core.physical_device, surface_composite);

        let surface_format =
            VulkanGraphicsSetup::choose_swapchain_format(&swapchain_support.formats);
        let present_mode =
            VulkanGraphicsSetup::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = VulkanGraphicsSetup::choose_swapchain_extent(
            &swapchain_support.capabilities,
            window_width,
            window_height,
        );

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_indices) =
            if core.queue_family.graphics_family != core.queue_family.present_family {
                (
                    vk::SharingMode::EXCLUSIVE,
                    vec![
                        core.queue_family.graphics_family.unwrap(),
                        core.queue_family.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            surface: surface_composite.surface,
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            image_array_layers: 1,
            ..Default::default()
        };

        let loader = ash::extensions::khr::Swapchain::new(&core.instance, &core.device);
        let swapchain = unsafe {
            loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create Swapchain!")
        };

        let images = unsafe {
            loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get Swapchain Images.")
        };

        SwapChainComposite {
            loader,
            swapchain,
            format: surface_format.format,
            extent,
            images,
            image_views: vec![],
            framebuffers: vec![],
        }
    }

    pub fn find_swapchain_support(
        physical_device: vk::PhysicalDevice,
        surface_composite: &SurfaceComposite,
    ) -> SwapChainSupportDetails {
        unsafe {
            let capabilities = surface_composite
                .loader
                .get_physical_device_surface_capabilities(
                    physical_device,
                    surface_composite.surface,
                )
                .expect("Failed to query for surface capabilities.");
            let formats = surface_composite
                .loader
                .get_physical_device_surface_formats(physical_device, surface_composite.surface)
                .expect("Failed to query for surface formats.");
            let present_modes = surface_composite
                .loader
                .get_physical_device_surface_present_modes(
                    physical_device,
                    surface_composite.surface,
                )
                .expect("Failed to query for surface present mode.");

            SwapChainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    fn choose_swapchain_format(
        available_formats: &Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        // check if list contains most widely used R8G8B8A8 format with nonlinear color space
        let selected_format = available_formats.iter().find(|format| {
            format.format == COLOR_FORMAT && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        });

        // return the first format from the list
        match selected_format {
            Some(f) => f.clone(),
            None => available_formats.first().unwrap().clone(),
        }
    }

    fn choose_swapchain_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        // prefer MAILBOX
        let selected_present_mode = available_present_modes
            .iter()
            .find(|present_mode| **present_mode == vk::PresentModeKHR::MAILBOX);

        // if not, use FIFO
        match selected_present_mode {
            Some(m) => *m,
            None => vk::PresentModeKHR::FIFO,
        }
    }

    fn choose_swapchain_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: window_width,
                height: window_height,
            }
        }
    }

    fn create_image_views(
        device: &ash::Device,
        swapchain_composite: &SwapChainComposite,
    ) -> Vec<vk::ImageView> {
        let mut swapchain_imageviews = vec![];

        for &image in swapchain_composite.images.iter() {
            let components = vk::ComponentMapping {
                ..Default::default()
            };
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
                ..Default::default()
            };
            let imageview_create_info = vk::ImageViewCreateInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                format: swapchain_composite.format,
                components,
                subresource_range,
                image,
                ..Default::default()
            };

            let imageview = unsafe {
                device
                    .create_image_view(&imageview_create_info, None)
                    .expect("Failed to create Image View!")
            };
            swapchain_imageviews.push(imageview);
        }

        swapchain_imageviews
    }

    fn create_render_pass(
        core: &VulkanCore,
        surface_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
    ) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: VulkanGraphicsSetup::find_depth_format(&core.instance, core.physical_device),
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_resolve = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format, // swapchain_format
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_resolve_ref = vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpasses = [vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            preserve_attachment_count: 0,
            p_resolve_attachments: &color_attachment_resolve_ref,
            ..Default::default()
        }];

        let render_pass_attachments =
            [color_attachment, depth_attachment, color_attachment_resolve];

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

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
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

    fn create_pipeline(
        core: &VulkanCore,
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
        descriptor_set_layout: vk::DescriptorSetLayout,
        msaa_samples: vk::SampleCountFlags,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let device = &core.device;
        let vert_shader_module =
            core.create_shader_module(include_bytes!("../../target/shaders/simple.vert.spv"));
        let frag_shader_module =
            core.create_shader_module(include_bytes!("../../target/shaders/simple.frag.spv"));

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
            rasterization_samples: msaa_samples,
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

    fn choose_msaa_samples(core: &VulkanCore) -> vk::SampleCountFlags {
        let physical_device_properties = unsafe {
            core.instance
                .get_physical_device_properties(core.physical_device)
        };

        let count = std::cmp::min(
            physical_device_properties
                .limits
                .framebuffer_color_sample_counts,
            physical_device_properties
                .limits
                .framebuffer_depth_sample_counts,
        );

        // if count.contains(vk::SampleCountFlags::TYPE_64) {
        //     return vk::SampleCountFlags::TYPE_64;
        // }
        // if count.contains(vk::SampleCountFlags::TYPE_32) {
        //     return vk::SampleCountFlags::TYPE_32;
        // }
        // if count.contains(vk::SampleCountFlags::TYPE_16) {
        //     return vk::SampleCountFlags::TYPE_16;
        // }
        // if count.contains(vk::SampleCountFlags::TYPE_8) {
        //     return vk::SampleCountFlags::TYPE_8;
        // }
        if count.contains(vk::SampleCountFlags::TYPE_4) {
            return vk::SampleCountFlags::TYPE_4;
        }
        if count.contains(vk::SampleCountFlags::TYPE_2) {
            return vk::SampleCountFlags::TYPE_2;
        }

        vk::SampleCountFlags::TYPE_1
    }

    fn create_color_resources(
        core: &VulkanCore,
        swapchain_extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::ImageView, vk::DeviceMemory) {
        let color_format = COLOR_FORMAT;
        let (color_image, color_image_memory) = core.create_image(
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            msaa_samples,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &core.physical_device_memory_properties,
        );
        let color_image_view =
            core.create_image_view(color_image, color_format, vk::ImageAspectFlags::COLOR, 1);

        (color_image, color_image_view, color_image_memory)
    }

    fn create_depth_resources(
        core: &VulkanCore,
        swapchain_extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::ImageView, vk::DeviceMemory) {
        let depth_format =
            VulkanGraphicsSetup::find_depth_format(&core.instance, core.physical_device);
        let (depth_image, depth_image_memory) = core.create_image(
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &core.physical_device_memory_properties,
        );
        let depth_image_view =
            core.create_image_view(depth_image, depth_format, vk::ImageAspectFlags::DEPTH, 1);

        (depth_image, depth_image_view, depth_image_memory)
    }

    fn find_depth_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> vk::Format {
        VulkanGraphicsSetup::find_supported_format(
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

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        image_views: &Vec<vk::ImageView>,
        color_image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
        swapchain_extent: &vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = vec![];

        for &image_view in image_views.iter() {
            let attachments = [color_image_view, depth_image_view, image_view];

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

    fn create_descriptor_pool(
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

    pub fn framebuffer_resized(&mut self, window_width: u32, window_height: u32) {
        self.window_width = window_width;
        self.window_height = window_height;
    }

    pub fn recreate_swapchain(&mut self) {
        let surface_composite = SurfaceComposite {
            loader: self.surface_composite.loader.clone(),
            surface: self.surface_composite.surface,
        };

        unsafe {
            self.core
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.cleanup_swapchain();

        self.swapchain_composite = VulkanGraphicsSetup::create_swapchain(
            &self.core,
            &surface_composite,
            self.window_width,
            self.window_height,
        );

        self.swapchain_composite.image_views =
            VulkanGraphicsSetup::create_image_views(&self.core.device, &self.swapchain_composite);
        self.render_pass = VulkanGraphicsSetup::create_render_pass(
            &self.core,
            self.swapchain_composite.format,
            self.msaa_samples,
        );
        let (graphics_pipeline, pipeline_layout) = VulkanGraphicsSetup::create_pipeline(
            &self.core,
            self.render_pass,
            self.swapchain_composite.extent,
            self.descriptor_set_layout,
            self.msaa_samples,
        );
        self.pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;

        let (color_image, color_image_view, color_image_memory) =
            VulkanGraphicsSetup::create_color_resources(
                &self.core,
                self.swapchain_composite.extent,
                self.msaa_samples,
            );
        self.color_image = color_image;
        self.color_image_view = color_image_view;
        self.color_image_memory = color_image_memory;
        let (depth_image, depth_image_view, depth_image_memory) =
            VulkanGraphicsSetup::create_depth_resources(
                &self.core,
                self.swapchain_composite.extent,
                self.msaa_samples,
            );
        self.depth_image = depth_image;
        self.depth_image_view = depth_image_view;
        self.depth_image_memory = depth_image_memory;

        self.swapchain_composite.framebuffers = VulkanGraphicsSetup::create_framebuffers(
            &self.core.device,
            self.render_pass,
            &self.swapchain_composite.image_views,
            self.color_image_view,
            self.depth_image_view,
            &self.swapchain_composite.extent,
        );
    }

    pub fn cleanup_swapchain(&self) {
        unsafe {
            let device = &self.core.device;
            device.destroy_image_view(self.color_image_view, None);
            device.destroy_image(self.color_image, None);
            device.free_memory(self.color_image_memory, None);
            device.destroy_image_view(self.depth_image_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_image_memory, None);

            for &framebuffer in self.swapchain_composite.framebuffers.iter() {
                device.destroy_framebuffer(framebuffer, None);
            }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            for &image_view in self.swapchain_composite.image_views.iter() {
                device.destroy_image_view(image_view, None);
            }
            self.swapchain_composite
                .loader
                .destroy_swapchain(self.swapchain_composite.swapchain, None);
        }
    }

    pub fn drop(&self) {
        unsafe {
            self.core
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.core
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.surface_composite
                .loader
                .destroy_surface(self.surface_composite.surface, None);
            self.core
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
