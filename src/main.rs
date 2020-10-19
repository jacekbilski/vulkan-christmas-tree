use winit::event_loop::EventLoop;

use vulkan::Vulkan;

mod vulkan;

pub mod fs;

// settings
const SCR_WIDTH: u32 = 1920;
const SCR_HEIGHT: u32 = 1080;

const APPLICATION_NAME: &'static str = "Vulkan Christmas Tree";

const CLEAR_VALUE: [f32; 4] = [0.015_7, 0., 0.360_7, 1.0];

fn main() {
    let event_loop = EventLoop::new();
    let window = init_window(&event_loop);

    let vulkan = Vulkan::new(window, APPLICATION_NAME, CLEAR_VALUE);
    vulkan.main_loop(event_loop);
}

fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
    winit::window::WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(winit::dpi::PhysicalSize::new(SCR_WIDTH, SCR_HEIGHT))
        .build(event_loop)
        .expect("Failed to create window.")
}
