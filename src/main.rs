use std::f32::consts::{FRAC_PI_8, TAU};

use winit::dpi::PhysicalPosition;
use winit::event::ElementState::Pressed;
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use vulkan::Vulkan;

use crate::fps_calculator::FpsCalculator;
use crate::scene::Scene;

mod vulkan;

mod coords;
mod fps_calculator;
mod fs;
mod mesh;
mod scene;

// settings
const SCR_WIDTH: u32 = 1920;
const SCR_HEIGHT: u32 = 1080;

const AUTO_ROTATION_SPEED_RAD_PER_SEC: f32 = TAU / 30.0;

const APPLICATION_NAME: &'static str = "Vulkan Christmas Tree";

fn main() {
    #[cfg(feature = "validation-layers")]
    println!("Running with validation layers enabled");
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
    let mut fps_calculator = FpsCalculator::new();
    let mut autorotate = false;
    let mut mouse_rotating = false;
    let mut last_cursor_position: PhysicalPosition<f64> = PhysicalPosition::new(0.0, 0.0);
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
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_vertically(angle_change, &mut vulkan);
            }
            VirtualKeyCode::Down => {
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_vertically(-angle_change, &mut vulkan);
            }
            VirtualKeyCode::Left => {
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_horizontally(-angle_change, &mut vulkan);
            }
            VirtualKeyCode::Right => {
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_horizontally(angle_change, &mut vulkan);
            }
            VirtualKeyCode::R => {
                autorotate = !autorotate;
            }
            _ => (),
        },
        Event::WindowEvent {
            event:
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                },
            ..
        } => {
            mouse_rotating = state == Pressed;
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            if mouse_rotating {
                let x_diff = position.x - last_cursor_position.x;
                let y_diff = position.y - last_cursor_position.y;

                let angle_change = FRAC_PI_8 / 128.;
                scene.rotate_camera_horizontally(-angle_change * x_diff as f32, &mut vulkan);
                scene.rotate_camera_vertically(angle_change * y_diff as f32, &mut vulkan);
            }
            last_cursor_position = position;
        }
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
            fps_calculator.tick();
            let last_frame_time_secs = fps_calculator.last_frame_time_secs();
            if autorotate {
                scene.rotate_camera_horizontally(
                    AUTO_ROTATION_SPEED_RAD_PER_SEC * last_frame_time_secs,
                    &mut vulkan,
                );
            }
            vulkan.draw_frame(last_frame_time_secs);
        }
        Event::LoopDestroyed => {
            vulkan.wait_device_idle();
        }
        _ => (),
    });
}
