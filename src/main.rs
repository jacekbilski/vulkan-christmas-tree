use std::f32::consts::FRAC_PI_8;
use std::time::Instant;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use vulkan::Vulkan;

use crate::scene::Scene;

mod vulkan;

mod coords;
mod fs;
mod mesh;
mod scene;

// settings
const SCR_WIDTH: u32 = 1920;
const SCR_HEIGHT: u32 = 1080;

const APPLICATION_NAME: &'static str = "Vulkan Christmas Tree";

fn main() {
    let event_loop = EventLoop::new();
    let window = init_window(&event_loop);
    let mut vulkan = Vulkan::new(&window, APPLICATION_NAME);
    let scene = Scene::setup(&mut vulkan, &window);
    main_loop(vulkan, window, scene, event_loop);
}

fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
    winit::window::WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(winit::dpi::PhysicalSize::new(SCR_WIDTH, SCR_HEIGHT))
        .build(event_loop)
        .expect("Failed to create window.")
}

fn main_loop(
    mut vulkan: vulkan::Vulkan,
    window: winit::window::Window,
    mut scene: Scene,
    event_loop: EventLoop<()>,
) {
    let mut ticker = Instant::now();
    let mut rotate = true;
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
            VirtualKeyCode::Up => {
                rotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_vertically(angle_change, &mut vulkan);
            }
            VirtualKeyCode::Down => {
                rotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_vertically(-angle_change, &mut vulkan);
            }
            VirtualKeyCode::Left => {
                rotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_horizontally(angle_change, &mut vulkan);
            }
            VirtualKeyCode::Right => {
                rotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_horizontally(-angle_change, &mut vulkan);
            }
            VirtualKeyCode::R => {
                rotate = !rotate;
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
            if rotate {
                let delta = ticker.elapsed().subsec_micros() as f32;
                scene.rotate_camera_horizontally(delta / 100_000.0, &mut vulkan);
            }
            vulkan.draw_frame();
            ticker = Instant::now();
        }
        Event::LoopDestroyed => {
            vulkan.wait_device_idle();
        }
        _ => (),
    });
}
