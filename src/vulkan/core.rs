use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
#[cfg(any(feature = "validation-layers", target_os = "windows"))]
use std::os::raw::c_void;
use std::ptr;

#[cfg(feature = "validation-layers")]
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::Surface;
#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::extensions::khr::{WaylandSurface, XlibSurface};
use ash::vk;
use ash::vk::PhysicalDeviceType;

use crate::vulkan::{QueueFamilyIndices, SurfaceComposite, VulkanGraphicsSetup};

const APPLICATION_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
const ENGINE_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
const VULKAN_API_VERSION: u32 = vk::make_api_version(0, 1, 1, 0);

#[derive(Clone)]
pub struct VulkanCore {
    _entry: ash::Entry,
    pub instance: ash::Instance,

    #[cfg(feature = "validation-layers")]
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    #[cfg(feature = "validation-layers")]
    debug_messenger: vk::DebugUtilsMessengerEXT,

    pub physical_device: vk::PhysicalDevice,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device: ash::Device,

    pub queue_family: QueueFamilyIndices,
    pub compute_queue: vk::Queue,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    transfer_queue: vk::Queue,
}

impl VulkanCore {
    pub fn new(window: &winit::window::Window, application_name: &str) -> (Self, SurfaceComposite) {
        let entry = unsafe { ash::Entry::new().unwrap() };
        let instance = VulkanCore::create_instance(&entry, application_name);
        #[cfg(feature = "validation-layers")]
        let (debug_utils_loader, debug_messenger) =
            VulkanCore::setup_debug_utils(&entry, &instance);
        let surface_composite = VulkanCore::create_surface(&entry, &instance, &window);
        let physical_device = VulkanCore::pick_physical_device(&instance, &surface_composite);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, queue_family) =
            VulkanCore::create_logical_device(&instance, physical_device, &surface_composite);
        let compute_queue =
            unsafe { device.get_device_queue(queue_family.compute_family.unwrap(), 0) };
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family.present_family.unwrap(), 0) };
        let transfer_queue =
            unsafe { device.get_device_queue(queue_family.transfer_family.unwrap(), 0) };
        (
            VulkanCore {
                _entry: entry,
                instance,

                #[cfg(feature = "validation-layers")]
                debug_utils_loader,
                #[cfg(feature = "validation-layers")]
                debug_messenger,

                physical_device,
                physical_device_memory_properties,

                device,
                queue_family,
                compute_queue,
                graphics_queue,
                present_queue,
                transfer_queue,
            },
            surface_composite,
        )
    }

    pub(crate) fn create_image(
        &self,
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
            self.device
                .create_image(&image_create_info, None)
                .expect("Failed to create Texture Image!")
        };

        let image_memory_requirement = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: image_memory_requirement.size,
            memory_type_index: VulkanCore::find_memory_type(
                image_memory_requirement.memory_type_bits,
                required_memory_properties,
                device_memory_properties,
            ),
            ..Default::default()
        };

        let image_memory = unsafe {
            self.device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate Texture Image memory!")
        };

        unsafe {
            self.device
                .bind_image_memory(image, image_memory, 0)
                .expect("Failed to bind Image Memmory!");
        }

        (image, image_memory)
    }

    pub(crate) fn create_image_view(
        &self,
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
            self.device
                .create_image_view(&imageview_create_info, None)
                .expect("Failed to create Image View!")
        }
    }

    pub fn create_data_buffer<T>(
        &self,
        command_pool: vk::CommandPool,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut T;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

            self.device.unmap_memory(staging_buffer_memory);
        }

        let (buffer, buffer_memory) = self.create_buffer(
            buffer_size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        self.copy_buffer(command_pool, staging_buffer, buffer, buffer_size);

        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_buffer_memory, None);
        }

        (buffer, buffer_memory)
    }

    pub(crate) fn create_buffer(
        &self,
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
            self.device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create Vertex Buffer")
        };

        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = VulkanCore::find_memory_type(
            mem_requirements.memory_type_bits,
            required_memory_properties,
            &self.physical_device_memory_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let buffer_memory = unsafe {
            self.device
                .allocate_memory(&allocate_info, None)
                .expect("Failed to allocate vertex buffer memory!")
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind Buffer");
        }

        (buffer, buffer_memory)
    }

    pub(crate) fn copy_buffer(
        &self,
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
            self.device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate Command Buffer")
        };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin Command Buffer");

            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }];

            self.device
                .cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);

            self.device
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
            self.device
                .queue_submit(self.transfer_queue, &submit_info, vk::Fence::null())
                .expect("Failed to Submit Queue.");
            self.device
                .queue_wait_idle(self.transfer_queue)
                .expect("Failed to wait Queue idle");

            self.device
                .free_command_buffers(command_pool, &command_buffers);
        }
    }

    pub(crate) fn create_shader_module(&self, shader_spv: &[u8]) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: shader_spv.len(),
            p_code: shader_spv.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            self.device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create Shader Module!")
        }
    }

    pub(crate) fn create_command_pool(&self, queue_family_index: u32) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            ..Default::default()
        };

        unsafe {
            self.device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create Command Pool!")
        }
    }

    pub(crate) fn create_semaphore(&self) -> vk::Semaphore {
        let semaphore_create_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };
        unsafe {
            self.device
                .create_semaphore(&semaphore_create_info, None)
                .expect("Failed to create Semaphore Object!")
        }
    }

    pub(crate) fn create_fence(&self) -> vk::Fence {
        let fence_create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        unsafe {
            self.device
                .create_fence(&fence_create_info, None)
                .expect("Failed to create Fence Object!")
        }
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

    fn create_instance(entry: &ash::Entry, application_name: &str) -> ash::Instance {
        let app_name = CString::new(application_name).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: APPLICATION_VERSION,
            p_engine_name: engine_name.as_ptr(),
            engine_version: ENGINE_VERSION,
            api_version: VULKAN_API_VERSION,
            ..Default::default()
        };

        let enabled_layer_raw_names: Vec<CString> = VulkanCore::required_layer_names()
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect();
        let enabled_layer_names: Vec<*const c_char> = enabled_layer_raw_names
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let enabled_extension_raw_names: Vec<CString> = VulkanCore::required_extension_names()
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let enabled_extension_names: Vec<*const c_char> = enabled_extension_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        #[cfg(feature = "validation-layers")]
        let debug_utils_messenger_ci = VulkanCore::build_messenger_create_info();

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            #[cfg(feature = "validation-layers")]
            p_next: &debug_utils_messenger_ci as *const vk::DebugUtilsMessengerCreateInfoEXT
                as *const c_void,
            enabled_layer_count: enabled_layer_names.len() as u32,
            pp_enabled_layer_names: enabled_layer_names.as_ptr(),
            enabled_extension_count: enabled_extension_names.len() as u32,
            pp_enabled_extension_names: enabled_extension_names.as_ptr(),
            ..Default::default()
        };

        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        instance
    }

    #[cfg(feature = "validation-layers")]
    fn setup_debug_utils(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);

        let messenger_ci = VulkanCore::build_messenger_create_info();

        let utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_ci, None)
                .expect("Debug Utils Callback")
        };

        (debug_utils_loader, utils_messenger)
    }

    #[cfg(feature = "validation-layers")]
    fn build_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
        vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(vulkan_debug_utils_callback),
            ..Default::default()
        }
    }

    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> SurfaceComposite {
        let surface = unsafe {
            use winit::platform::unix::WindowExtUnix;

            if window.wayland_surface() != None {
                let wayland_surface = window.wayland_surface().unwrap();
                let wayland_display = window.wayland_display().unwrap();
                let wayland_create_info = vk::WaylandSurfaceCreateInfoKHR {
                    surface: wayland_surface,
                    display: wayland_display,
                    ..Default::default()
                };
                let wayland_surface_loader = WaylandSurface::new(entry, instance);
                wayland_surface_loader
                    .create_wayland_surface(&wayland_create_info, None)
                    .expect("Failed to create surface.")
            } else {
                let x11_window = window.xlib_window().unwrap();
                let x11_display = window.xlib_display().unwrap();
                let x11_create_info = vk::XlibSurfaceCreateInfoKHR {
                    window: x11_window,
                    dpy: x11_display as *mut vk::Display,
                    ..Default::default()
                };
                let xlib_surface_loader = XlibSurface::new(entry, instance);
                xlib_surface_loader
                    .create_xlib_surface(&x11_create_info, None)
                    .expect("Failed to create surface.")
            }
        };
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

        SurfaceComposite {
            loader: surface_loader,
            surface,
        }
    }

    #[cfg(target_os = "windows")]
    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> SurfaceComposite {
        let surface = unsafe {
            use winapi::shared::windef::HWND;
            use winapi::um::libloaderapi::GetModuleHandleW;
            use winit::platform::windows::WindowExtWindows;

            let hwnd = window.hwnd() as HWND;
            let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
            let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
                s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: Default::default(),
                hinstance,
                hwnd: hwnd as *const c_void,
            };
            let loader = Win32Surface::new(entry, instance);
            loader
                .create_win32_surface(&win32_create_info, None)
                .expect("Failed to create surface.")
        };
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

        SurfaceComposite {
            loader: surface_loader,
            surface,
        }
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_composite: &SurfaceComposite,
    ) -> vk::PhysicalDevice {
        let physical_devices: Vec<vk::PhysicalDevice> = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate Physical Devices!")
        };

        let result = physical_devices.iter().find(|physical_device| {
            VulkanCore::is_physical_device_suitable(instance, **physical_device, &surface_composite)
        });

        match result {
            None => panic!("Failed to find a suitable GPU!"),
            Some(physical_device) => *physical_device,
        }
    }

    fn is_physical_device_suitable(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_composite: &SurfaceComposite,
    ) -> bool {
        let indices = VulkanCore::find_queue_family(instance, physical_device, &surface_composite);
        let is_queue_family_supported = indices.is_complete();

        let is_device_extension_supported =
            VulkanCore::check_device_extension_support(instance, physical_device);

        let is_swapchain_supported = if is_device_extension_supported {
            let swapchain_support =
                VulkanGraphicsSetup::find_swapchain_support(physical_device, surface_composite);
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        } else {
            false
        };

        let is_discrete_gpu = unsafe {
            let props = instance.get_physical_device_properties(physical_device);
            props.device_type == PhysicalDeviceType::DISCRETE_GPU
        };

        return is_queue_family_supported
            && is_device_extension_supported
            && is_swapchain_supported
            && is_discrete_gpu;
    }

    fn find_queue_family(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_composite: &SurfaceComposite,
    ) -> QueueFamilyIndices {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut queue_family_indices = QueueFamilyIndices::new();

        let mut index: u32 = 0;
        for queue_family in queue_families.iter() {
            if queue_family.queue_count > 0 {
                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    queue_family_indices.graphics_family = Some(index);
                    queue_family_indices.transfer_family = Some(index);
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    queue_family_indices.compute_family = Some(index);
                    queue_family_indices.transfer_family = Some(index);
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    queue_family_indices.transfer_family = Some(index);
                }

                let is_present_support = unsafe {
                    surface_composite
                        .loader
                        .get_physical_device_surface_support(
                            physical_device,
                            index as u32,
                            surface_composite.surface,
                        )
                        .unwrap()
                };
                if is_present_support {
                    queue_family_indices.present_family = Some(index);
                }
            }

            if queue_family_indices.is_complete() {
                break;
            }

            index += 1;
        }

        queue_family_indices
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Failed to get device extension properties.")
        };

        let mut available_extension_names = vec![];

        for extension in available_extensions.iter() {
            let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }
                .to_str()
                .unwrap()
                .to_owned();
            available_extension_names.push(extension_name);
        }

        let mut required_extensions = HashSet::new();
        for extension in VulkanCore::required_device_extensions().iter() {
            required_extensions.insert(extension.to_string());
        }

        for extension_name in available_extension_names.iter() {
            required_extensions.remove(extension_name);
        }

        return required_extensions.is_empty();
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_composite: &SurfaceComposite,
    ) -> (ash::Device, QueueFamilyIndices) {
        let indices = VulkanCore::find_queue_family(instance, physical_device, surface_composite);

        let queue_priorities: [f32; 1] = [1.0];
        let queue_create_infos = [vk::DeviceQueueCreateInfo {
            queue_family_index: indices.graphics_family.unwrap(),
            queue_count: queue_priorities.len() as u32,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let physical_device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let enabled_layer_raw_names: Vec<CString> = VulkanCore::required_layer_names()
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect();
        let enabled_layer_names: Vec<*const c_char> = enabled_layer_raw_names
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        let enabled_extension_raw_names: Vec<CString> = VulkanCore::required_device_extensions()
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect();
        let enabled_extension_names: Vec<*const c_char> = enabled_extension_raw_names
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: enabled_layer_names.len() as u32,
            pp_enabled_layer_names: enabled_layer_names.as_ptr(),
            p_enabled_features: &physical_device_features,
            enabled_extension_count: enabled_extension_names.len() as u32,
            pp_enabled_extension_names: enabled_extension_names.as_ptr(),
            ..Default::default()
        };

        let device: ash::Device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical Device!")
        };

        (device, indices)
    }

    fn required_layer_names() -> Vec<&'static str> {
        vec![
            #[cfg(feature = "validation-layers")]
            "VK_LAYER_KHRONOS_validation",
        ]
    }

    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
    fn required_extension_names() -> Vec<&'static str> {
        vec![
            Surface::name().to_str().unwrap(),
            XlibSurface::name().to_str().unwrap(),
            WaylandSurface::name().to_str().unwrap(),
            #[cfg(feature = "validation-layers")]
            DebugUtils::name().to_str().unwrap(),
        ]
    }

    #[cfg(all(windows))]
    fn required_extension_names() -> Vec<&'static str> {
        vec![
            Surface::name().to_str().unwrap(),
            Win32Surface::name().to_str().unwrap(),
            #[cfg(feature = "validation-layers")]
            DebugUtils::name().to_str().unwrap(),
        ]
    }

    fn required_device_extensions() -> Vec<&'static str> {
        vec![ash::extensions::khr::Swapchain::name().to_str().unwrap()]
    }

    pub fn drop(&self) {
        unsafe {
            self.device.destroy_device(None);
            #[cfg(feature = "validation-layers")]
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// the callback function used in Debug Utils.
#[cfg(feature = "validation-layers")]
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}
