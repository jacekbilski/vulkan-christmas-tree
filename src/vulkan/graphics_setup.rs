use ash::version::DeviceV1_0;
use ash::vk;

use crate::vulkan::core::VulkanCore;
use crate::vulkan::{SurfaceComposite, SwapChainComposite, Vulkan};

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

    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,

    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,

    pub window_width: u32,
    pub window_height: u32,
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
        let render_pass = Vulkan::create_render_pass(&core, swapchain_composite.format);
        let graphics_descriptor_set_layout =
            Vulkan::create_graphics_descriptor_set_layout(&core.device);
        let (graphics_pipeline, graphics_pipeline_layout) = Vulkan::create_graphics_pipeline(
            &core.device,
            render_pass,
            swapchain_composite.extent,
            graphics_descriptor_set_layout,
        );
        let (depth_image, depth_image_view, depth_image_memory) =
            Vulkan::create_depth_resources(&core, swapchain_composite.extent);
        swapchain_composite.framebuffers = Vulkan::create_framebuffers(
            &core.device,
            render_pass,
            &swapchain_composite.image_views,
            depth_image_view,
            &swapchain_composite.extent,
        );
        let graphics_command_pool =
            Vulkan::create_command_pool(&core.device, core.queue_family.graphics_family.unwrap());
        let graphics_descriptor_pool =
            Vulkan::create_graphics_descriptor_pool(&core.device, swapchain_composite.images.len());

        VulkanGraphicsSetup {
            core,

            surface_composite,
            swapchain_composite,

            render_pass,
            descriptor_set_layout: graphics_descriptor_set_layout,
            pipeline_layout: graphics_pipeline_layout,
            pipeline: graphics_pipeline,

            depth_image,
            depth_image_view,
            depth_image_memory,

            command_pool: graphics_command_pool,
            descriptor_pool: graphics_descriptor_pool,

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
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
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
        self.render_pass = Vulkan::create_render_pass(&self.core, self.swapchain_composite.format);
        let (graphics_pipeline, pipeline_layout) = Vulkan::create_graphics_pipeline(
            &self.core.device,
            self.render_pass,
            self.swapchain_composite.extent,
            self.descriptor_set_layout,
        );
        self.pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;

        let (depth_image, depth_image_view, depth_image_memory) =
            Vulkan::create_depth_resources(&self.core, self.swapchain_composite.extent);
        self.depth_image = depth_image;
        self.depth_image_view = depth_image_view;
        self.depth_image_memory = depth_image_memory;

        self.swapchain_composite.framebuffers = Vulkan::create_framebuffers(
            &self.core.device,
            self.render_pass,
            &self.swapchain_composite.image_views,
            self.depth_image_view,
            &self.swapchain_composite.extent,
        );
    }

    pub fn cleanup_swapchain(&self) {
        unsafe {
            let device = &self.core.device;
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

    pub fn drop(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.surface_composite
                .loader
                .destroy_surface(self.surface_composite.surface, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}
