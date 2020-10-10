use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, XlibSurface};
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use ash::vk::DebugUtilsMessengerCreateInfoEXT;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

// settings
const SCR_WIDTH: u32 = 1920;
const SCR_HEIGHT: u32 = 1080;

const VALIDATION_LAYER_NAME: &'static str = "VK_LAYER_KHRONOS_validation";

const WINDOW_TITLE: &'static str = "Vulkan Christmas Tree";

const APPLICATION_VERSION: u32 = vk::make_version(0, 1, 0);
const ENGINE_VERSION: u32 = vk::make_version(0, 1, 0);
const VULKAN_API_VERSION: u32 = vk::make_version(1, 2, 148);

struct App {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl App {
    pub fn new() -> Self {
        let entry = ash::Entry::new().unwrap();
        let instance = App::create_instance(&entry);
        let (debug_utils_loader, debug_messenger) = App::setup_debug_utils(&entry, &instance);

        App {
            _entry: entry,
            instance,
            debug_utils_loader,
            debug_messenger,
        }
    }

    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::PhysicalSize::new(SCR_WIDTH, SCR_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.")
    }

    fn create_instance(entry: &ash::Entry) -> ash::Instance {

        let layer_properties = entry
            .enumerate_instance_layer_properties()
            .expect("Failed to enumerate Instance Layers Properties");
        println!("Found {} layers", layer_properties.len());
        for layer_property in layer_properties.iter() {
            println!("Found layer: {:?}", layer_property);
        }

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

        // let layer_names: Vec<*const c_char> = required_layer_names();
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
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
            .pfn_user_callback(Some(vulkan_debug_utils_callback))
            .build()
    }

    fn draw_frame(&mut self) {
        // Drawing will be here
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) {
        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            virtual_keycode: Some(virtual_code),
                            state: ElementState::Pressed,
                            ..
                        },
                        ..
                    },
                    .. } => {
                    match virtual_code {
                        VirtualKeyCode::Escape => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => ()
                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    self.draw_frame();
                }
                _ => ()
            }
        });
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = App::init_window(&event_loop);

    let app = App::new();
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
