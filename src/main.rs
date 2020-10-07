use std::ffi::CString;
use std::os::raw::c_char;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, XlibSurface};
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

// settings
const SCR_WIDTH: u32 = 1920;
const SCR_HEIGHT: u32 = 1080;

const WINDOW_TITLE: &'static str = "Vulkan Christmas Tree";

const APPLICATION_VERSION: u32 = vk::make_version(0, 1, 0);
const ENGINE_VERSION: u32 = vk::make_version(0, 1, 0);
const VULKAN_API_VERSION: u32 = vk::make_version(1, 2, 148);

struct App {
    _entry: ash::Entry,
    instance: ash::Instance,
}

impl App {
    pub fn new() -> Self {
        let entry = ash::Entry::new().unwrap();
        let instance = App::create_instance(&entry);

        App {
            _entry: entry,
            instance,
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
        let app_name = CString::new(WINDOW_TITLE).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(APPLICATION_VERSION)
            .engine_name(engine_name.as_c_str())
            .engine_version(ENGINE_VERSION)
            .api_version(VULKAN_API_VERSION)
            .build();

        let extension_names = required_extension_names();

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .build();

        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        instance
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

fn required_extension_names() -> Vec<*const c_char> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}
