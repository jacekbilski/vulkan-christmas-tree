use std::time::Instant;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use vulkan::Vulkan;

use crate::mesh::Mesh;
use crate::vulkan::Vertex;

mod vulkan;

mod fs;
mod mesh;

// settings
const SCR_WIDTH: u32 = 1920;
const SCR_HEIGHT: u32 = 1080;

const APPLICATION_NAME: &'static str = "Vulkan Christmas Tree";

const CLEAR_VALUE: [f32; 4] = [0.015_7, 0., 0.360_7, 1.0];

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5],
        color: [1.0, 1.0, 1.0],
    },
];
const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];

fn main() {
    let event_loop = EventLoop::new();
    let window = init_window(&event_loop);
    let mesh = Mesh {
        vertices: Vec::from(VERTICES_DATA),
        indices: Vec::from(INDICES_DATA),
    };
    let vulkan = Vulkan::new(&window, APPLICATION_NAME, CLEAR_VALUE, mesh);
    main_loop(vulkan, window, event_loop);
}

fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
    winit::window::WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(winit::dpi::PhysicalSize::new(SCR_WIDTH, SCR_HEIGHT))
        .build(event_loop)
        .expect("Failed to create window.")
}

fn main_loop(mut vulkan: vulkan::Vulkan, window: winit::window::Window, event_loop: EventLoop<()>) {
    let mut ticker = Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            vulkan.wait_device_idle();
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
                vulkan.wait_device_idle();
                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        },
        Event::WindowEvent {
            event: WindowEvent::Resized(_new_size),
            ..
        } => {
            vulkan.wait_device_idle();
            vulkan.framebuffer_resized(window.inner_size().width, window.inner_size().height);
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_window_id) => {
            let delta = ticker.elapsed().subsec_micros();
            vulkan.draw_frame(delta as f32 / 1000_000.0_f32);
            ticker = Instant::now();
        }
        Event::LoopDestroyed => {
            vulkan.wait_device_idle();
        }
        _ => (),
    });
}
