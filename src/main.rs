use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, XlibSurface};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use ash::vk::DebugUtilsMessengerCreateInfoEXT;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

// settings
const SCR_WIDTH: u32 = 1920;
const SCR_HEIGHT: u32 = 1080;

const VALIDATION_LAYER_NAME: &'static str = "VK_LAYER_KHRONOS_validation";
const DEVICE_EXTENSIONS: [&'static str; 1] = ["VK_KHR_swapchain"];

const WINDOW_TITLE: &'static str = "Vulkan Christmas Tree";

const APPLICATION_VERSION: u32 = vk::make_version(0, 1, 0);
const ENGINE_VERSION: u32 = vk::make_version(0, 1, 0);
const VULKAN_API_VERSION: u32 = vk::make_version(1, 2, 154);

struct QueueFamilyIndices {
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

struct SurfaceComposite {
    loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
}

struct SwapChainComposite {
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
}

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

struct App {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    _physical_device: vk::PhysicalDevice,
    device: ash::Device,

    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,

    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    _swapchain_images: Vec<vk::Image>,
    _swapchain_format: vk::Format,
    _swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
}

impl App {
    pub fn new(window: &Window) -> Self {
        let entry = ash::Entry::new().unwrap();
        let instance = App::create_instance(&entry);
        let surface_composite = App::create_surface(&entry, &instance, &window);
        let (debug_utils_loader, debug_messenger) = App::setup_debug_utils(&entry, &instance);
        let physical_device = App::pick_physical_device(&instance, &surface_composite);
        let (device, family_indices) =
            App::create_logical_device(&instance, physical_device, &surface_composite);
        let swapchain_composite = App::create_swapchain(
            &instance,
            &device,
            physical_device,
            // &window,
            &surface_composite,
            &family_indices,
        );
        let swapchain_imageviews = App::create_image_views(
            &device,
            swapchain_composite.format,
            &swapchain_composite.images,
        );

        let graphics_queue =
            unsafe { device.get_device_queue(family_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(family_indices.present_family.unwrap(), 0) };

        App {
            _entry: entry,
            instance,
            surface_loader: surface_composite.loader,
            surface: surface_composite.surface,
            debug_utils_loader,
            debug_messenger,

            _physical_device: physical_device,
            device,

            _graphics_queue: graphics_queue,
            _present_queue: present_queue,

            swapchain_loader: swapchain_composite.loader,
            swapchain: swapchain_composite.swapchain,
            _swapchain_format: swapchain_composite.format,
            _swapchain_images: swapchain_composite.images,
            _swapchain_extent: swapchain_composite.extent,
            swapchain_imageviews,
        }
    }

    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::PhysicalSize::new(SCR_WIDTH, SCR_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.")
    }

    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> SurfaceComposite {
        let surface = unsafe {
            use winit::platform::unix::WindowExtUnix;

            let x11_window = window.xlib_window().unwrap();
            let x11_display = window.xlib_display().unwrap();
            let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                .window(x11_window)
                .dpy(x11_display as *mut vk::Display)
                .build();
            let xlib_surface_loader = XlibSurface::new(entry, instance);
            xlib_surface_loader
                .create_xlib_surface(&x11_create_info, None)
                .expect("Failed to create surface.")
        };
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

        SurfaceComposite {
            loader: surface_loader,
            surface,
        }
    }

    fn create_instance(entry: &ash::Entry) -> ash::Instance {
        let app_name = CString::new(WINDOW_TITLE).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(APPLICATION_VERSION)
            .engine_name(engine_name.as_c_str())
            .engine_version(ENGINE_VERSION)
            .api_version(VULKAN_API_VERSION)
            .build();

        let enable_layer_raw_names: Vec<CString> = required_layer_names()
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let enable_layer_names: Vec<*const i8> = enable_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let extension_names: Vec<*const c_char> = required_extension_names();

        let mut messenger_ci = App::build_messenger_create_info();

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .push_next(&mut messenger_ci)
            .enabled_layer_names(&enable_layer_names)
            .enabled_extension_names(&extension_names)
            .build();

        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        instance
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
            App::is_physical_device_suitable(instance, **physical_device, &surface_composite)
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
        let indices = App::find_queue_family(instance, physical_device, &surface_composite);
        let is_queue_family_supported = indices.is_complete();

        let is_device_extension_supported =
            App::check_device_extension_support(instance, physical_device);

        let is_swapchain_supported = if is_device_extension_supported {
            let swapchain_support = App::find_swapchain_support(physical_device, surface_composite);
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        } else {
            false
        };

        return is_queue_family_supported
            && is_device_extension_supported
            && is_swapchain_supported;
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
        for extension in DEVICE_EXTENSIONS.iter() {
            required_extensions.insert(extension.to_string());
        }

        for extension_name in available_extension_names.iter() {
            required_extensions.remove(extension_name);
        }

        return required_extensions.is_empty();
    }

    fn find_swapchain_support(
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

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_composite: &SurfaceComposite,
        queue_family: &QueueFamilyIndices,
    ) -> SwapChainComposite {
        let swapchain_support = App::find_swapchain_support(physical_device, surface_composite);

        let surface_format = App::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = App::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = App::choose_swapchain_extent(&swapchain_support.capabilities);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
            if queue_family.graphics_family != queue_family.present_family {
                (
                    vk::SharingMode::EXCLUSIVE,
                    2,
                    vec![
                        queue_family.graphics_family.unwrap(),
                        queue_family.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface_composite.surface,
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            queue_family_index_count,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            image_array_layers: 1,
        };

        let loader = ash::extensions::khr::Swapchain::new(instance, device);
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

    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: SCR_WIDTH,
                height: SCR_HEIGHT,
            }
        }
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_composite: &SurfaceComposite,
    ) -> (ash::Device, QueueFamilyIndices) {
        let indices = App::find_queue_family(instance, physical_device, surface_composite);

        let queue_priorities: [f32; 1] = [1.0];
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(indices.graphics_family.unwrap())
            .queue_priorities(&queue_priorities)
            .build();

        let physical_device_features = vk::PhysicalDeviceFeatures::builder().build();

        let enable_layer_raw_names: Vec<CString> = required_layer_names()
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let enable_layer_names: Vec<*const i8> = enable_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();
        let enable_extension_names = [
            ash::extensions::khr::Swapchain::name().as_ptr(), // currently just enable the Swapchain extension.
        ];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[queue_create_info])
            .enabled_layer_names(&enable_layer_names)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&enable_extension_names)
            .build();

        let device: ash::Device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical Device!")
        };

        (device, indices)
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

    fn create_image_views(
        device: &ash::Device,
        surface_format: vk::Format,
        images: &Vec<vk::Image>,
    ) -> Vec<vk::ImageView> {
        let mut swapchain_imageviews = vec![];

        for &image in images.iter() {
            let components = vk::ComponentMapping::builder().build();
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();
            let imageview_create_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format)
                .components(components)
                .subresource_range(subresource_range)
                .image(image)
                .build();

            let imageview = unsafe {
                device
                    .create_image_view(&imageview_create_info, None)
                    .expect("Failed to create Image View!")
            };
            swapchain_imageviews.push(imageview);
        }

        swapchain_imageviews
    }

    fn setup_debug_utils(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);

        let messenger_ci = App::build_messenger_create_info();

        let utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_ci, None)
                .expect("Debug Utils Callback")
        };

        (debug_utils_loader, utils_messenger)
    }

    fn build_messenger_create_info() -> DebugUtilsMessengerCreateInfoEXT {
        vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                    // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
                    // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback))
            .build()
    }

    fn draw_frame(&mut self) {
        // Drawing will be here
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(virtual_code),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    },
                ..
            } => match virtual_code {
                VirtualKeyCode::Escape => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_window_id) => {
                self.draw_frame();
            }
            _ => (),
        });
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            for &imageview in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(imageview, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = App::init_window(&event_loop);

    let app = App::new(&window);
    app.main_loop(event_loop, window);
}

/// the callback function used in Debug Utils.
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

fn required_layer_names() -> Vec<&'static str> {
    vec![VALIDATION_LAYER_NAME]
}

fn required_extension_names() -> Vec<*const c_char> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}
